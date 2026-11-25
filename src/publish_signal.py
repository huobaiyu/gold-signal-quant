# -*- coding: utf-8 -*-
"""
发布月度信号到 signals/ 目录（公开 track record）并推送到 GitHub。
- 读取 results/signal_latest.json，归档为 signals/<target_month>.json
- 回填：历史信号对应月份的实际金价已公布时，计算实际涨跌与是否命中
- 重建 signals/README.md 汇总表（含累计命中率）
- git commit + push（仅在有变更时）
用法：python src/publish_signal.py
"""
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNALS = ROOT / "signals"
RESULT = ROOT / "results" / "signal_latest.json"


def fetch_gold() -> pd.Series:
    raw = subprocess.run(
        ["curl", "-sL", "--max-time", "120",
         "https://datahub.io/core/gold-prices/r/monthly.csv"],
        capture_output=True, text=True, check=True, timeout=150).stdout
    d = pd.read_csv(io.StringIO(raw))
    d["Date"] = pd.to_datetime(d["Date"])
    s = d.set_index("Date")["Price"].astype(float)
    s.index = s.index.to_period("M").to_timestamp()
    return s.sort_index()


def main():
    if not RESULT.exists():
        print("未找到 results/signal_latest.json，请先运行 monthly_signal.py")
        return 1
    SIGNALS.mkdir(exist_ok=True)
    sig = json.loads(RESULT.read_text())
    target = sig["target_month"]  # e.g. "2026-11"
    sig["published_at"] = datetime.now().isoformat(timespec="seconds")
    (SIGNALS / f"{target}.json").write_text(json.dumps(sig, ensure_ascii=False, indent=2))
    print(f"已归档信号 signals/{target}.json（{sig['signal']}，上涨概率 {sig['prob_up']:.1%}）")

    # 回填历史信号的实际结果
    px = fetch_gold()
    records = []
    hits, settled = 0, 0
    for f in sorted(SIGNALS.glob("????-??.json")):
        rec = json.loads(f.read_text())
        tm = rec["target_month"]
        if "actual_return_pct" not in rec:
            t = pd.Period(tm, freq="M").to_timestamp()
            prev = (pd.Period(tm, freq="M") - 1).to_timestamp()
            if t in px.index and prev in px.index:
                ret = float(px.loc[t] / px.loc[prev] - 1)
                rec["actual_return_pct"] = round(ret * 100, 2)
                rec["actual_direction"] = "上涨" if ret > 0 else "下跌"
                rec["hit"] = bool((rec["signal"] == "看多") == (ret > 0))
                f.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
                print(f"回填 {tm}：实际{'上涨' if ret > 0 else '下跌'} {ret:+.2%}，"
                      f"{'命中' if rec['hit'] else '未命中'}")
        if "hit" in rec:
            settled += 1
            hits += int(rec["hit"])
        records.append(rec)

    # 汇总表
    lines = [
        "# 月度信号追踪记录（Public Track Record）",
        "",
        "每月 4 日（北京时间）由自动化任务生成并提交，commit 时间戳即发布时间，不可篡改。",
        "信号针对**当月**金价方向（看多/观望），次月数据公布后回填实际结果。",
        "",
        "| 信号月份 | 发布时间 | 信号 | 上涨概率 | 建议仓位 | 实际涨跌 | 是否命中 |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        actual = f"{rec['actual_return_pct']:+.2f}%" if "actual_return_pct" in rec else "待公布"
        hit = ("✅" if rec["hit"] else "❌") if "hit" in rec else "—"
        pos = f"{rec['position_pct']:.0%}" if "position_pct" in rec else "—"
        pub = rec.get("published_at", "")[:10]
        lines.append(f"| {rec['target_month']} | {pub} | {rec['signal']} | "
                     f"{rec['prob_up']:.1%} | {pos} | {actual} | {hit} |")
    lines.append("")
    if settled:
        lines.append(f"**累计已结算 {settled} 期，命中 {hits} 期，命中率 {hits/settled:.1%}**"
                     f"（回测参考：方向准确率约 56%）")
    else:
        lines.append("**暂无已结算信号**（回测参考：方向准确率约 56%）")
    lines += ["", "> 仅供研究参考，不构成投资建议。", ""]
    (SIGNALS / "README.md").write_text("\n".join(lines))
    print("已更新 signals/README.md")

    # git commit + push
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args],
                              capture_output=True, text=True)
    git("add", "signals", "results/signal_latest.json")
    if not git("status", "--porcelain").stdout.strip():
        print("无变更，跳过提交")
        return 0
    msg = f"signal {target}: {sig['signal']} (prob_up={sig['prob_up']:.1%})"
    r = git("commit", "-m", msg)
    if r.returncode != 0:
        print("commit 失败:", r.stderr.strip() or r.stdout.strip())
        return 1
    r = git("push")
    print("已推送到 GitHub" if r.returncode == 0 else f"push 失败: {r.stderr.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
