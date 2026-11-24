# -*- coding: utf-8 -*-
"""
Improved approach vs naive replication.
- Richer price features (multi-scale momentum, MA ratios, realized vol)
- Walk-forward expanding-window retraining (no look-ahead, adapts to regime)
- Probability threshold: trade only when confident
- Benchmarks: buy&hold, 12m time-series momentum
Evaluation: full walk-forward OOS 2010-2026 AND the last-48-month window.
"""
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/monthly-processed.csv", parse_dates=["Date"]).set_index("Date").asfreq("MS")
px = df["Price"].astype(float)
r1 = px.pct_change()

feat = pd.DataFrame(index=px.index)
for k in (1, 2, 3, 6, 12):
    feat[f"ret_{k}m"] = px.pct_change(k)
for w in (3, 6, 12):
    feat[f"ma_ratio_{w}"] = px / px.rolling(w).mean() - 1
    feat[f"vol_{w}"] = r1.rolling(w).std()
feat["mom_12_1"] = px.shift(1).pct_change(12)  # classic 12-1 momentum

next_ret = r1.shift(-1)
label = (next_ret > 0).astype(int)
data = feat.join(pd.DataFrame({"next_ret": next_ret, "label": label})).dropna().loc["1975-01-01":]
FCOLS = [c for c in data.columns if c not in ("next_ret", "label")]

# ---------- walk-forward: train 1975..Y-1, predict year Y, for Y=2010..2026 ----------
def walk_forward(model_fn, threshold=None):
    preds = {}
    years = range(2010, 2027)
    for y in years:
        tr = data.loc[:f"{y-1}-12-01"]
        te = data.loc[f"{y}-01-01":f"{y}-12-01"]
        if len(te) == 0 or len(tr) < 100:
            continue
        sc = StandardScaler().fit(tr[FCOLS])
        m = model_fn()
        m.fit(sc.transform(tr[FCOLS]), tr["label"])
        p = m.predict_proba(sc.transform(te[FCOLS]))[:, 1]
        for dt, prob in zip(te.index, p):
            preds[dt] = prob
    s = pd.Series(preds).sort_index()
    if threshold is None:
        sig = (s > 0.5).astype(float)
    else:
        lo, hi = threshold
        sig = pd.Series(np.where(s > hi, 1.0, np.where(s < lo, 0.0, np.nan)), index=s.index)
        sig = sig.ffill().fillna(0.0)  # keep previous position when uncertain
    return s, sig

def perf(sig: pd.Series, cost=0.001):
    d = data.loc[sig.index]
    r = d["next_ret"].values
    s = sig.values
    changes = np.abs(np.diff(np.concatenate([[0.0], s])))
    strat = s * r - changes * cost
    eq = np.cumprod(1 + strat)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    sharpe = float(ann / (np.std(strat) * np.sqrt(12)))
    acc = float(((s > 0.5).astype(int) == d["label"].values).mean())
    turnover = float(changes.sum())
    return dict(total=float(eq[-1] - 1), ann=float(ann), mdd=mdd, sharpe=sharpe,
                dir_acc=acc, turnover=turnover), eq

def buyhold(idx):
    d = data.loc[idx]
    r = d["next_ret"].values
    eq = np.cumprod(1 + r)
    n = len(r)
    ann = eq[-1] ** (12 / n) - 1
    mdd = float(np.min(eq / np.maximum.accumulate(eq) - 1))
    return dict(total=float(eq[-1]-1), ann=float(ann), mdd=mdd,
                sharpe=float(ann/(np.std(r)*np.sqrt(12)))), eq

# models
def mk_logit(): return LogisticRegression(max_iter=3000, C=0.5)
def mk_gbm(): return GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                learning_rate=0.05, subsample=0.8, random_state=0)

prob_logit, sig_logit = walk_forward(mk_logit)
prob_gbm, sig_gbm = walk_forward(mk_gbm)
prob_ens = (prob_logit + prob_gbm) / 2
sig_ens = (prob_ens > 0.5).astype(float)
_, sig_ens_thr = walk_forward(mk_logit, threshold=None)  # placeholder
sig_ens_thr = pd.Series(np.where(prob_ens > 0.56, 1.0, np.where(prob_ens < 0.44, 0.0, np.nan)),
                        index=prob_ens.index).ffill().fillna(0.0)

# time-series momentum benchmark: long if 12m momentum > 0
mom_sig = (data["mom_12_1"].loc[prob_ens.index] > 0).astype(float)

results = {}
curves = {}
for name, sig in [("WF_Logistic", sig_logit), ("WF_GBM", sig_gbm),
                  ("WF_Ensemble", sig_ens), ("WF_Ensemble_conf", sig_ens_thr),
                  ("TS_Momentum", mom_sig)]:
    m, eq = perf(sig)
    results[name] = m
    curves[name] = eq
bh, eq_bh = buyhold(prob_ens.index)
results["BuyHold"] = bh
curves["BuyHold"] = eq_bh

print("=== Full walk-forward OOS 2010-2026 ===")
print(pd.DataFrame(results).T.round(3).to_string())

# ---------- same strategies restricted to last 48 months ----------
cut = pd.Timestamp("2022-11-01")
print("\n=== Last 48 months (2022.11-2026.09) ===")
sub = {}
for name, sig in [("WF_Logistic", sig_logit), ("WF_GBM", sig_gbm),
                  ("WF_Ensemble", sig_ens), ("WF_Ensemble_conf", sig_ens_thr),
                  ("TS_Momentum", mom_sig)]:
    m, _ = perf(sig.loc[cut:])
    sub[name] = m
m, _ = buyhold(prob_ens.loc[cut:].index)
sub["BuyHold"] = m
print(pd.DataFrame(sub).T.round(3).to_string())

# direction accuracy of ensemble in the last 48 months
d = data.loc[prob_ens.loc[cut:].index]
acc48 = float(((prob_ens.loc[cut:] > 0.5).astype(int) == d["label"]).mean())
print(f"\nEnsemble direction accuracy last 48m: {acc48:.3f}")

pd.DataFrame({"date": prob_ens.index, "prob_up": prob_ens.values,
              "sig_ens": sig_ens.values, "sig_conf": sig_ens_thr.values,
              "sig_mom": mom_sig.values}).to_csv("results/walkforward_signals.csv", index=False)
pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).to_csv("results/walkforward_curves.csv")
with open("results/walkforward_metrics.json", "w") as f:
    json.dump({"full": results, "last48": sub, "ens_acc_last48": acc48}, f, indent=2)
print("saved walkforward_signals.csv / walkforward_curves.csv / walkforward_metrics.json")
