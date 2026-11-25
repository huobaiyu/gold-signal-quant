# -*- coding: utf-8 -*-
"""
仓位管理回测：0/1 开关 vs 概率强度 × 波动率目标的连续仓位。
数据/特征/模型与 backtest_flows.py 完全一致（宏观+COT 生产版），仅改仓位规则。
"""
import json
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
dgs10, cpi = load_fred(WS / "fred_dgs10.csv"), load_fred(WS / "fred_cpi.csv")
unrate, vix, dxy = load_fred(WS / "fred_unrate.csv"), load_fred(WS / "fred_vix.csv"), load_fred(WS / "fred_dxy.csv")
cot = pd.read_csv(WS / "cot_gold_weekly.csv", parse_dates=["date"]).set_index("date")
cot_m = cot[["nc_net_pct_oi"]].resample("MS").last()
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
exog = pd.DataFrame({
    "real_rate": real_rate, "d_real_rate_3m": real_rate.diff(3),
    "cpi_yoy": cpi_yoy, "unrate": unrate, "d_unrate_12m": unrate.diff(12),
    "vix": vix, "d_vix_3m": vix.diff(3), "dxy_ret_12m": dxy.pct_change(12),
    "cot_net_pct_oi": cot_m["nc_net_pct_oi"], "d_cot_3m": cot_m["nc_net_pct_oi"].diff(3),
}).shift(1)

data = feat.join(exog)
data["next_ret"] = r1.shift(-1)
data["label"] = (data["next_ret"] > 0).astype(int)
FCOLS = list(feat.columns) + list(exog.columns)
data = data.dropna(subset=FCOLS).loc["1975-01-01":]

probs = {}
for y in range(2012, 2027):
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
realized_vol_12m = data["vol_12"] * np.sqrt(12)  # 年化已实现波动率

# ---- 仓位规则 ----
pos_binary = (prob > 0.5).astype(float)
# 概率强度：p<=0.5 → 0；p>=0.65 → 满仓；线性过渡
pos_prob = ((prob - 0.50) / 0.15).clip(0, 1)
# 概率强度 × 波动率目标（目标年化波动 12%，无杠杆，上限 1）
vol_scalar = (0.12 / realized_vol_12m.loc[prob.index]).clip(upper=1.0)
pos_vol = (pos_prob * vol_scalar).clip(0, 1)


def perf(pos, cost=0.001):
    d = data.loc[pos.index].dropna(subset=["next_ret"])
    pos = pos.loc[d.index]
    r = d["next_ret"].values
    s = pos.values
    ch = np.abs(np.diff(np.concatenate([[0.0], s])))
    strat = s * r - ch * cost
    eq = np.cumprod(1 + strat)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    vol = float(np.std(strat) * np.sqrt(12))
    return dict(total=round(float(eq[-1] - 1), 3), ann=round(float(ann), 3),
                mdd=round(mdd, 3), vol=round(vol, 3),
                sharpe=round(float(ann / vol) if vol > 0 else 0, 3),
                avg_pos=round(float(np.mean(s)), 3),
                turnover=round(float(ch.sum()), 1)), pd.Series(eq, index=d.index)


out, curves = {}, {}
for name, pos in [("binary_0_1", pos_binary), ("prob_scaled", pos_prob), ("prob_vol_target", pos_vol)]:
    m, eq = perf(pos)
    m48, _ = perf(pos.loc[pd.Timestamp("2022-11-01"):])
    out[name] = {"oos2012": m, "last48": m48}
    curves[name] = eq

print(pd.DataFrame({k: v["oos2012"] for k, v in out.items()}).to_string())
print()
print(pd.DataFrame({k: v["last48"] for k, v in out.items()}).to_string())

pd.DataFrame(curves).to_csv(WS / "position_curves.csv")
with open(WS / "position_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved position_curves.csv / position_metrics.json")
