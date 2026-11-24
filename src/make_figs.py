# -*- coding: utf-8 -*-
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial Unicode MS", "PingFang SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- fig 1: replication accuracy vs claimed ----------
rep = json.load(open("results/backtest_replica_metrics.json"))
acc = rep["accuracy"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

ax = axes[0]
names = ["原项目宣称\nMLP(日频2012-20)", "永远看涨\n(朴素基线)", "线性回归\n(复现)", "逻辑回归\n(复现)", "MLP\n(复现,10次均值)"]
vals = [0.80, acc["naive_always_up"], acc["linear_regression"],
        acc["logistic_regression"], acc["mlp_mean_acc"]]
colors = ["#999", "#888", "#4C72B0", "#55A868", "#C44E52"]
bars = ax.bar(names, vals, color=colors)
ax.axhline(0.5, ls="--", c="k", lw=0.8, alpha=0.5)
ax.text(4.4, 0.505, "抛硬币 50%", fontsize=8, ha="right")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.1%}", ha="center", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_ylabel("下月涨跌方向预测准确率")
ax.set_title("复现回测：近48个月(2022.11–2026.09) 样本外准确率")
ax.tick_params(axis="x", labelsize=8)

ax = axes[1]
mlp_runs = acc["mlp_runs"]
ax.bar(range(1, 11), mlp_runs, color="#C44E52", alpha=0.8)
ax.axhline(np.mean(mlp_runs), color="k", ls="--", lw=1, label=f"均值 {np.mean(mlp_runs):.1%}")
ax.axhline(acc["mlp_train_acc"], color="#4C72B0", ls=":", lw=1.2,
           label=f"同一MLP训练集内准确率 {acc['mlp_train_acc']:.1%}")
ax.set_xlabel("MLP 随机种子编号（同一数据训练10次）")
ax.set_ylabel("样本外准确率")
ax.set_title("MLP 样本外表现极不稳定 → 严重过拟合")
ax.legend(fontsize=8)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig("figures/fig1_replication.png")
plt.close()

# ---------- fig 2: equity curves ----------
bt = pd.read_csv("results/backtest_replica.csv", parse_dates=["date"])
wf = pd.read_csv("results/walkforward_curves.csv")
sig_dates = pd.read_csv("results/walkforward_signals.csv", parse_dates=["date"])["date"]
wf.index = sig_dates.values[:len(wf)]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
ax = axes[0]
ax.plot(bt["date"], bt["eq_bh"], label="买入持有", c="#888", lw=1.5)
ax.plot(bt["date"], bt["eq_lr"], label="线性回归信号(复现)", c="#4C72B0", lw=1.2)
ax.plot(bt["date"], bt["eq_mlp"], label="MLP信号(复现,10次投票)", c="#C44E52", lw=1.2)
ax.set_title("近48个月策略净值（复现版, 含0.1%换手成本）")
ax.legend(fontsize=8)
ax.set_ylabel("净值（起点=1）")
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(wf.index, wf["BuyHold"], label="买入持有", c="#888", lw=1.5)
ax.plot(wf.index, wf["WF_Ensemble"], label="改进版：滚动集成模型", c="#C44E52", lw=1.4)
ax.plot(wf.index, wf["WF_Logistic"], label="改进版：滚动逻辑回归", c="#4C72B0", lw=1.4)
ax.plot(wf.index, wf["TS_Momentum"], label="12个月动量", c="#55A868", lw=1.2)
ax.set_yscale("log")
ax.set_title("改进版：2010–2026 滚动样本外净值（对数轴）")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("figures/fig2_equity.png")
plt.close()
print("figs saved")
