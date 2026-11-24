# -*- coding: utf-8 -*-
"""
Macro-enhanced walk-forward backtest.
Macro features are LAGGED 1 month (publication lag -> no look-ahead).
Variants:
  price_only : price features only (baseline from previous step)
  macro75    : + real rate (DGS10 - CPI YoY), d(real rate), CPI YoY, UNRATE, d(UNRATE)   [1975+]
  macro_full : macro75 + VIX, d(VIX), DXY 12m change                                     [2006+]
Evaluation: expanding-window yearly retrain; OOS windows 2000-2026 (macro75) and
2012-2026 (macro_full), plus last 48 months for comparability.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def load_fred(path):
    s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].astype(float)
    s.index = s.index.to_period("M").to_timestamp()
    return s

ws = str(Path(__file__).resolve().parent.parent)  # 项目根目录
px = pd.read_csv(f"{ws}/data/monthly-processed.csv", parse_dates=["Date"]).set_index("Date")["Price"].astype(float)
px.index = px.index.to_period("M").to_timestamp()
r1 = px.pct_change()

dgs10 = load_fred(f"{ws}/data/fred_dgs10.csv")
cpi = load_fred(f"{ws}/data/fred_cpi.csv")
unrate = load_fred(f"{ws}/data/fred_unrate.csv")
vix = load_fred(f"{ws}/data/fred_vix.csv")
dxy = load_fred(f"{ws}/data/fred_dxy.csv")

feat = pd.DataFrame(index=px.index)
for k in (1, 2, 3, 6, 12):
    feat[f"ret_{k}m"] = px.pct_change(k)
for w in (3, 6, 12):
    feat[f"ma_ratio_{w}"] = px / px.rolling(w).mean() - 1
    feat[f"vol_{w}"] = r1.rolling(w).std()
feat["mom_12_1"] = px.shift(1).pct_change(12)
PRICE_COLS = list(feat.columns)

cpi_yoy = cpi.pct_change(12) * 100
real_rate = dgs10 - cpi_yoy
macro75 = pd.DataFrame({
    "real_rate": real_rate,
    "d_real_rate_3m": real_rate.diff(3),
    "cpi_yoy": cpi_yoy,
    "unrate": unrate,
    "d_unrate_12m": unrate.diff(12),
}).shift(1)  # publication lag
macro_full_extra = pd.DataFrame({
    "vix": vix,
    "d_vix_3m": vix.diff(3),
    "dxy_ret_12m": dxy.pct_change(12),
}).shift(1)

next_ret = r1.shift(-1)
label = (next_ret > 0).astype(int)

variants = {
    "price_only": feat,
    "macro75": feat.join(macro75),
    "macro_full": feat.join(macro75).join(macro_full_extra),
}

def run_variant(F, start_oos):
    data = F.join(pd.DataFrame({"next_ret": next_ret, "label": label})).dropna()
    FCOLS = [c for c in data.columns if c not in ("next_ret", "label")]
    probs = {}
    for y in range(start_oos, 2027):
        tr = data.loc[:f"{y-1}-12-01"]
        te = data.loc[f"{y}-01-01":f"{y}-12-01"]
        if len(te) == 0 or len(tr) < 120:
            continue
        sc = StandardScaler().fit(tr[FCOLS])
        Xtr, ytr = sc.transform(tr[FCOLS]), tr["label"].values
        m1 = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr)
        m2 = GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                        learning_rate=0.05, subsample=0.8, random_state=0).fit(Xtr, ytr)
        Xte = sc.transform(te[FCOLS])
        p = (m1.predict_proba(Xte)[:, 1] + m2.predict_proba(Xte)[:, 1]) / 2
        for dt, pv in zip(te.index, p):
            probs[dt] = pv
    prob = pd.Series(probs).sort_index()
    sig = (prob > 0.5).astype(float)
    return data, prob, sig

def perf(data, sig, cost=0.001):
    d = data.loc[sig.index]
    r = d["next_ret"].values
    s = sig.values
    ch = np.abs(np.diff(np.concatenate([[0.0], s])))
    strat = s * r - ch * cost
    eq = np.cumprod(1 + strat)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    sharpe = float(ann / (np.std(strat) * np.sqrt(12)))
    acc = float(((s > 0.5).astype(int) == d["label"].values).mean())
    return dict(total=round(float(eq[-1]-1), 3), ann=round(float(ann), 3),
                mdd=round(mdd, 3), sharpe=round(sharpe, 3), dir_acc=round(acc, 3)), eq

def buyhold(data, idx):
    r = data.loc[idx, "next_ret"].values
    eq = np.cumprod(1 + r)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    return dict(total=round(float(eq[-1]-1), 3), ann=round(float(ann), 3),
                mdd=round(mdd, 3), sharpe=round(float(ann/(np.std(r)*np.sqrt(12))), 3)), eq

out, curves = {}, {}
common_idx = {}
data_price, prob_price, sig_price = run_variant(variants["price_only"], 2000)
data_m75, prob_m75, sig_m75 = run_variant(variants["macro75"], 2000)
data_mf, prob_mf, sig_mf = run_variant(variants["macro_full"], 2012)

# common OOS window for fair comparison: 2012-2026 (all three available)
cut12 = pd.Timestamp("2012-01-01")
for name, data, sig in [("price_only", data_price, sig_price),
                        ("macro75", data_m75, sig_m75),
                        ("macro_full", data_mf, sig_mf)]:
    m, eq = perf(data, sig.loc[cut12:])
    out[name] = m
    curves[name] = pd.Series(eq, index=sig.loc[cut12:].index)
m, eq = buyhold(data_mf, data_mf.loc[cut12:].index)
out["BuyHold"] = m
curves["BuyHold"] = pd.Series(eq, index=data_mf.loc[cut12:].index)

print("=== OOS 2012-2026 (all variants, 0.1% cost) ===")
print(pd.DataFrame(out).T.to_string())

# last 48 months
cut48 = pd.Timestamp("2022-11-01")
sub = {}
for name, data, sig in [("price_only", data_price, sig_price),
                        ("macro75", data_m75, sig_m75),
                        ("macro_full", data_mf, sig_mf)]:
    m, _ = perf(data, sig.loc[cut48:])
    sub[name] = m
m, _ = buyhold(data_mf, data_mf.loc[cut48:].index)
sub["BuyHold"] = m
print("\n=== Last 48 months (2022.11-2026.09) ===")
print(pd.DataFrame(sub).T.to_string())

pd.DataFrame(curves).to_csv(f"{ws}/results/macro_curves.csv")
pd.DataFrame({"date": prob_mf.index, "prob_up_macro": prob_mf.values,
              "sig_macro": sig_mf.values}).to_csv(f"{ws}/results/macro_signals.csv", index=False)
with open(f"{ws}/results/macro_metrics.json", "w") as f:
    json.dump({"oos2012": out, "last48": sub}, f, indent=2)
print("saved macro_curves.csv / macro_signals.csv / macro_metrics.json")
