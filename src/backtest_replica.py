# -*- coding: utf-8 -*-
"""
Backtest of github.com/SHUAIBAO-AI/Predict-Gold-Future-Price-Quant methodology
on monthly gold prices (1833-2026.10), testing the last 48 months (2022.11-2026.10).

The original project: daily data 2012-2020, 62 exogenous features (gold, VIX, DXY,
US Treasury), Linear Regression -> 25% accuracy; MLP binary classifier -> "80%".
Our attachment has ONLY monthly gold price, so we replicate the *methodology* with
price-derived features (lagged returns), same model classes, same "10 runs average".
"""
import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

# ---------- data ----------
df = pd.read_csv("data/monthly-processed.csv", parse_dates=["Date"])
df = df.set_index("Date").asfreq("MS")
px = df["Price"].astype(float)
ret1 = px.pct_change()

# Gold price was officially fixed until ~1968/1971; training on the peg era is noise.
# Use free-float era for model training, but keep full series for reference.
START_TRAIN = "1975-01-01"

N_LAG = 12  # 12 lagged monthly returns as features (price-only analogue of "all dimensions")

feat = pd.DataFrame(index=px.index)
for k in range(1, N_LAG + 1):
    feat[f"ret_lag{k}"] = ret1.shift(k)

# label: direction of NEXT month return
next_ret = ret1.shift(-1)
label = (next_ret > 0).astype(int)

data = feat.join(pd.DataFrame({"next_ret": next_ret, "label": label})).dropna()
data = data.loc[START_TRAIN:]

# ---------- split: train <= 2022-10, test = last 48 months (2022-11 .. 2026-10) ----------
TEST_START = "2022-11-01"
train = data.loc[:pd.Timestamp(TEST_START) - pd.offsets.MonthBegin(1)]
test = data.loc[TEST_START:]
print(f"train: {train.index[0].date()} .. {train.index[-1].date()}  n={len(train)}")
print(f"test : {test.index[0].date()} .. {test.index[-1].date()}  n={len(test)}")

Xtr, ytr = train[[c for c in data.columns if c.startswith("ret_lag")]].values, train["label"].values
Xte, yte = test[[c for c in data.columns if c.startswith("ret_lag")]].values, test["label"].values

scaler = StandardScaler().fit(Xtr)  # fit on train only (no leakage)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

results = {}

# ---------- Model 0: naive baselines ----------
# always up
results["naive_always_up"] = float((yte == 1).mean())
# momentum: next month = last month direction
mom_pred = (test["ret_lag1"].values > 0).astype(int)
results["naive_momentum"] = float((mom_pred == yte).mean())

# ---------- Model 1: Linear Regression (repo: predict next price, derive direction) ----------
# repo regressed next-day price on features; direction accuracy was 25%
lr = LinearRegression().fit(Xtr_s, train["next_ret"].values)
lr_pred_ret = lr.predict(Xte_s)
lr_dir = (lr_pred_ret > 0).astype(int)
results["linear_regression"] = float((lr_dir == yte).mean())

# ---------- Model 2: Logistic Regression (linear classifier, fair linear benchmark) ----------
logit = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
results["logistic_regression"] = float(logit.score(Xte_s, yte))

# ---------- Model 3: MLP binary classifier (repo's "80%" model), 10 runs average ----------
mlp_accs, mlp_preds = [], []
for seed in range(10):
    mlp = MLPClassifier(hidden_layer_sizes=(N_LAG, N_LAG), activation="relu",
                        max_iter=2000, random_state=seed)
    mlp.fit(Xtr_s, ytr)
    acc = mlp.score(Xte_s, yte)
    mlp_accs.append(acc)
    mlp_preds.append(mlp.predict(Xte_s))
results["mlp_runs"] = [round(a, 4) for a in mlp_accs]
results["mlp_mean_acc"] = float(np.mean(mlp_accs))
mlp_vote = (np.mean(mlp_preds, axis=0) > 0.5).astype(int)
results["mlp_ensemble_acc"] = float((mlp_vote == yte).mean())

# in-sample accuracy of one MLP (to check overfit / reproduce repo-style number)
mlp_in = MLPClassifier(hidden_layer_sizes=(N_LAG, N_LAG), max_iter=2000, random_state=0).fit(Xtr_s, ytr)
results["mlp_train_acc"] = float(mlp_in.score(Xtr_s, ytr))
results["logistic_train_acc"] = float(logit.score(Xtr_s, ytr))

print(json.dumps(results, indent=2, ensure_ascii=False))

# ---------- trading simulation (long/flat, monthly, 0.1% cost per position change) ----------
def backtest(signal: np.ndarray, test_df: pd.DataFrame, cost=0.001):
    r = test_df["next_ret"].values
    sig = signal.astype(float)
    changes = np.abs(np.diff(np.concatenate([[0.0], sig])))
    strat = sig * r - changes * cost
    eq = np.cumprod(1 + strat)
    bh = np.cumprod(1 + r)
    def stats(x):
        n = len(x)
        ann = x[-1] ** (12 / n) - 1
        mdd = float(np.min(x / np.maximum.accumulate(x) - 1))
        vol = np.std(strat) * np.sqrt(12)
        sharpe = ann / vol if vol > 0 else np.nan
        return dict(total=float(x[-1] - 1), ann=float(ann), mdd=mdd, sharpe=float(sharpe))
    return stats(eq), stats(bh), eq, bh

mlp_bt, bh_bt, eq_mlp, eq_bh = backtest(mlp_vote, test)
lr_bt, _, eq_lr, _ = backtest(lr_dir, test)
logit_bt, _, eq_logit, _ = backtest(logit.predict(Xte_s), test)
print("MLP strategy:", json.dumps(mlp_bt, indent=2))
print("LR  strategy:", json.dumps(lr_bt, indent=2))
print("LOG strategy:", json.dumps(logit_bt, indent=2))
print("Buy&Hold   :", json.dumps(bh_bt, indent=2))

# save artifacts for plotting
out = pd.DataFrame({
    "date": test.index, "actual_dir": yte, "mlp_vote": mlp_vote,
    "lr_dir": lr_dir, "logit_dir": logit.predict(Xte_s),
    "next_ret": test["next_ret"].values,
    "eq_mlp": eq_mlp, "eq_lr": eq_lr, "eq_logit": eq_logit, "eq_bh": eq_bh,
})
out.to_csv("results/backtest_replica.csv", index=False)
with open("results/backtest_replica_metrics.json", "w") as f:
    json.dump({"accuracy": results, "strategy": {"mlp": mlp_bt, "lr": lr_bt,
               "logit": logit_bt, "buy_hold": bh_bt}}, f, indent=2, ensure_ascii=False)
print("saved backtest_replica.csv / backtest_replica_metrics.json")
