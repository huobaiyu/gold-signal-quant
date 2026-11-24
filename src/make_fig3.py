# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ws = str(Path(__file__).resolve().parent.parent)  # 项目根目录
curves = pd.read_csv(f"{ws}/results/macro_curves.csv", index_col=0, parse_dates=True)
mm = json.load(open(f"{ws}/results/macro_metrics.json"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

ax = axes[0]
ax.plot(curves.index, curves["BuyHold"], label="买入持有", c="#999", lw=1.5)
ax.plot(curves.index, curves["price_only"], label="纯价格特征（滚动集成）", c="#4C72B0", lw=1.3)
ax.plot(curves.index, curves["macro_full"], label="宏观增强版（实际利率/美元/VIX…）", c="#C44E52", lw=1.4)
ax.set_title("样本外净值 2012–2026（滚动重训, 含成本, 宏观特征滞后1月）")
ax.legend(fontsize=8)
ax.set_ylabel("净值（起点=1）")
ax.grid(alpha=0.3)

ax = axes[1]
labels = {"price_only": "纯价格", "macro75": "宏观75版", "macro_full": "宏观增强版", "BuyHold": "买入持有"}
x = range(len(labels))
sharpes = [mm["oos2012"][k]["sharpe"] for k in labels]
mdds = [abs(mm["oos2012"][k]["mdd"]) for k in labels]
w = 0.38
b1 = ax.bar([i - w/2 for i in x], sharpes, w, label="夏普比率", color="#4C72B0")
b2 = ax.bar([i + w/2 for i in x], mdds, w, label="最大回撤(绝对值)", color="#DD8452")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02, f"{b.get_height():.2f}",
            ha="center", fontsize=8)
ax.set_xticks(list(x))
ax.set_xticklabels(labels.values(), fontsize=9)
ax.set_title("2012–2026 样本外：夏普 vs 最大回撤")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{ws}/figures/fig3_macro.png")
print("saved fig3_macro.png")
