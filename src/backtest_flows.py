# -*- coding: utf-8 -*-
"""
资金与情绪因子（CFTC 非商业持仓）接入后的滚动样本外回测。
与 backtest_macro.py 完全相同的流程/超参，仅新增 COT 特征，保证可比性。
"""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

WS = Path("/Users/mac/Documents/kimi/tasks/2026-11-24/14-52-56-7456557d")


def load_fred(path):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].astype(float)
    s.index = s.index.to_period("M").to_timestamp()
    return s


px = pd.read_csv(WS / "monthly-processed.csv", parse_dates=["Date"]).set_index("Date")["Price"].astype(float)
px.index = px.index.to_period("M").to_timestamp()
r1 = px.pct_change()

dgs10 = load_fred(WS / "fred_dgs10.csv")
cpi = load_fred(WS / "fred_cpi.csv")
unrate = load_fred(WS / "fred_unrate.csv")
vix = load_fred(WS / "fred_vix.csv")
dxy = load_fred(WS / "fred_dxy.csv")

cot = pd.read_csv(WS / "cot_gold_weekly.csv", parse_dates=["date"]).set_index("date")
cot_m = cot[["nc_net_pct_oi"]].resample("MS").last()  # 月末最后一周持仓
cot_m.index = cot_m.index.to_period("M").to_timestamp()

feat = pd.DataFrame(index=px.index)
for k in (1, 2, 3, 6, 12):
    feat[f"ret_{k}m"] = px.pct_change(k)
for w in (3, 6, 12):
    feat[f"ma_ratio_{w}"] = px / px.rolling(w).mean() - 1
    feat[f"vol_{w}"] = r1.rolling(w).std()
feat["mom_12_1"] = px.shift(1).pct_change(12)

cpi_yoy = cpi.pct_change(12) * 100
real_rate = dgs10 - cpi_yoy
macro = pd.DataFrame({
    "real_rate": real_rate, "d_real_rate_3m": real_rate.diff(3),
    "cpi_yoy": cpi_yoy, "unrate": unrate, "d_unrate_12m": unrate.diff(12),
    "vix": vix, "d_vix_3m": vix.diff(3), "dxy_ret_12m": dxy.pct_change(12),
})
flows = pd.DataFrame({
    "cot_net_pct_oi": cot_m["nc_net_pct_oi"],
    "d_cot_3m": cot_m["nc_net_pct_oi"].diff(3),
})
# 所有外生特征统一滞后 1 个月（发布延迟，COT 周报也有 3 天延迟，月度口径下保守处理）
exog = macro.join(flows).shift(1)

next_ret = r1.shift(-1)
data = feat.join(exog)
data["next_ret"] = next_ret
data["label"] = (next_ret > 0).astype(int)
FCOLS = list(feat.columns) + list(exog.columns)
data = data.dropna(subset=FCOLS).loc["1975-01-01":]


def walk_forward(start_oos):
    probs = {}
    for y in range(start_oos, 2027):
        tr = data.loc[:f"{y-1}-12-01"].dropna(subset=["next_ret"])
        te = data.loc[f"{y}-01-01":f"{y}-12-01"]
        if len(te) == 0 or len(tr) < 120:
            continue
        sc = StandardScaler().fit(tr[FCOLS])
        Xtr, ytr = sc.transform(tr[FCOLS]), tr["label"].values
        m1 = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr)
        m2 = GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                        learning_rate=0.05, subsample=0.8, random_state=0).fit(Xtr, ytr)
        p = (m1.predict_proba(sc.transform(te[FCOLS]))[:, 1]
             + m2.predict_proba(sc.transform(te[FCOLS]))[:, 1]) / 2
        for dt, pv in zip(te.index, p):
            probs[dt] = pv
    prob = pd.Series(probs).sort_index()
    return prob, (prob > 0.5).astype(float)


def perf(sig, cost=0.001):
    d = data.loc[sig.index].dropna(subset=["next_ret"])
    sig = sig.loc[d.index]
    r = d["next_ret"].values
    s = sig.values
    ch = np.abs(np.diff(np.concatenate([[0.0], s])))
    strat = s * r - ch * cost
    eq = np.cumprod(1 + strat)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    acc = float(((s > 0.5).astype(int) == d["label"].values).mean())
    return dict(total=round(float(eq[-1] - 1), 3), ann=round(float(ann), 3),
                mdd=round(mdd, 3), sharpe=round(float(ann / (np.std(strat) * np.sqrt(12))), 3),
                dir_acc=round(acc, 3)), pd.Series(eq, index=d.index)


prob, sig = walk_forward(2012)
m_full, eq_full = perf(sig)
cut48 = pd.Timestamp("2022-11-01")
m_48, _ = perf(sig.loc[cut48:])
print("=== macro+COT flows, OOS 2012-2026 ===")
print(json.dumps(m_full, indent=2))
print("=== last 48 months ===")
print(json.dumps(m_48, indent=2))

# 保存信号与净值供绘图
eq_full.to_frame("eq_flows").to_csv(WS / "flows_curve.csv")
with open(WS / "flows_metrics.json", "w") as f:
    json.dump({"oos2012": m_full, "last48": m_48}, f, indent=2)
print("saved flows_curve.csv / flows_metrics.json")
