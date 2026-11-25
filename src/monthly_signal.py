# -*- coding: utf-8 -*-
"""
黄金月度多空信号生成脚本（定时任务用）
- 宏观数据：FRED 免密钥接口（fredgraph.csv），全自动更新
- 金价历史：monthly-processed.csv 为主干（1833-2026.10），
  gold_monthly_local.csv（Date,Price）为增量补充，由任务每次运行时维护
- 模型：逻辑回归 + 梯度提升集成，全样本（1975+）训练，与回测验证过的
  macro_full 变体一致；宏观特征滞后 1 个月（发布延迟，无前视偏差）
- 输出：signal_latest.json + 控制台中文摘要
用法：python monthly_signal.py
"""
import io
import json
import sys
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

WS = str(Path(__file__).resolve().parent.parent)  # 项目根目录（data/ 与 results/ 所在）
UA = {"User-Agent": "Mozilla/5.0"}

FRED_IDS = {
    "dgs10": "DGS10",      # 10年期美债名义收益率（日频→月均）
    "cpi": "CPIAUCSL",     # CPI（月频）
    "unrate": "UNRATE",    # 失业率（月频）
    "vix": "VIXCLS",       # VIX（日频→月均）
    "dxy": "DTWEXBGS",     # 广义美元指数（日频→月均）
}


def fetch_fred(series_id: str, retries: int = 3) -> pd.Series:
    import subprocess
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd=1970-01-01"
    last_err = None
    raw = None
    for _ in range(retries):
        try:
            raw = subprocess.run(["curl", "-s", "--max-time", "120", url],
                                 capture_output=True, text=True, check=True, timeout=150).stdout
            if raw and raw.splitlines()[0].lower().startswith(("date", "observation_date")):
                break
            last_err = f"unexpected content: {raw[:100]}"
            raw = None
        except Exception as e:  # noqa: BLE001
            last_err = e
    if raw is None:
        raise RuntimeError(f"FRED 下载失败 {series_id}: {last_err}")
    d = pd.read_csv(io.StringIO(raw))
    d.columns = ["date", "value"]
    d["date"] = pd.to_datetime(d["date"])
    d["value"] = pd.to_numeric(d["value"], errors="coerce")
    s = d.dropna().set_index("date")["value"]
    m = s.resample("MS").mean().dropna()
    m.index = m.index.to_period("M").to_timestamp()
    return m


def load_gold() -> pd.Series:
    """优先从 datahub.io 拉取最新月度金价（1833至今，与回测数据同源）；
    失败则退回本地 monthly-processed.csv 主干 + gold_monthly_local.csv 增量。"""
    import subprocess
    try:
        raw = subprocess.run(
            ["curl", "-sL", "--max-time", "120",
             "https://datahub.io/core/gold-prices/r/monthly.csv"],
            capture_output=True, text=True, check=True, timeout=150).stdout
        d = pd.read_csv(io.StringIO(raw))
        if not str(d.columns[0]).startswith("Date"):
            raise ValueError("unexpected header")
        d["Date"] = pd.to_datetime(d["Date"])
        s = d.set_index("Date")["Price"].astype(float)
        s.index = s.index.to_period("M").to_timestamp()
        print(f"  金价历史：datahub.io 在线更新，最新 {s.index[-1].strftime('%Y-%m')}")
        return s.sort_index()
    except Exception as e:  # noqa: BLE001
        print(f"  datahub.io 拉取失败（{e}），退回本地文件")
    base = pd.read_csv(f"{WS}/data/monthly-processed.csv", parse_dates=["Date"]).set_index("Date")["Price"].astype(float)
    base.index = base.index.to_period("M").to_timestamp()
    try:
        ext = pd.read_csv(f"{WS}/data/gold_monthly_local.csv", parse_dates=["Date"]).set_index("Date")["Price"].astype(float)
        ext.index = ext.index.to_period("M").to_timestamp()
        base = base[~base.index.isin(ext.index)]
        base = pd.concat([base, ext]).sort_index()
    except FileNotFoundError:
        pass
    return base


def load_cot() -> pd.Series:
    """读取 CFTC 黄金非商业净持仓（占未平仓比）；若缓存超过35天未更新，
    下载本年度 CFTC 文件增量合并。"""
    import zipfile
    path = Path(WS) / "data" / "cot_gold_weekly.csv"
    cot = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    if (pd.Timestamp.now() - cot.index[-1]).days > 35:
        try:
            year = pd.Timestamp.now().year
            url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
            raw = subprocess.run(["curl", "-sL", "--max-time", "90", url],
                                 capture_output=True, timeout=100).stdout
            z = zipfile.ZipFile(io.BytesIO(raw))
            df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
            gold = df[df["Market and Exchange Names"].str.strip()
                      == "GOLD - COMMODITY EXCHANGE INC."]
            cols = {"As of Date in Form YYYY-MM-DD": "date",
                    "Open Interest (All)": "oi",
                    "Noncommercial Positions-Long (All)": "nc_long",
                    "Noncommercial Positions-Short (All)": "nc_short",
                    "Commercial Positions-Long (All)": "c_long",
                    "Commercial Positions-Short (All)": "c_short"}
            gold = gold.rename(columns=cols)[list(cols.values())]
            gold["date"] = pd.to_datetime(gold["date"])
            gold = gold.set_index("date")
            cot = pd.concat([cot[~cot.index.isin(gold.index)], gold]).sort_index()
            for c in ("oi", "nc_long", "nc_short", "c_long", "c_short"):
                cot[c] = pd.to_numeric(cot[c], errors="coerce")
            cot["nc_net"] = cot["nc_long"] - cot["nc_short"]
            cot["nc_net_pct_oi"] = cot["nc_net"] / cot["oi"]
            cot.to_csv(path)
            print(f"  COT 持仓：已更新至 {cot.index[-1].date()}")
        except Exception as e:  # noqa: BLE001
            print(f"  COT 持仓更新失败（{e}），使用缓存")
    return cot["nc_net_pct_oi"].resample("MS").last()


def main():
    fetched_at = datetime.now().isoformat(timespec="seconds")
    print("正在更新宏观数据（FRED）...")
    fred = {k: fetch_fred(v) for k, v in FRED_IDS.items()}
    for k, s in fred.items():
        print(f"  {k}: 最新 {s.index[-1].strftime('%Y-%m')} = {s.iloc[-1]:.2f}")

    px = load_gold()
    r1 = px.pct_change()

    feat = pd.DataFrame(index=px.index)
    for k in (1, 2, 3, 6, 12):
        feat[f"ret_{k}m"] = px.pct_change(k)
    for w in (3, 6, 12):
        feat[f"ma_ratio_{w}"] = px / px.rolling(w).mean() - 1
        feat[f"vol_{w}"] = r1.rolling(w).std()
    feat["mom_12_1"] = px.shift(1).pct_change(12)

    cpi_yoy = fred["cpi"].pct_change(12) * 100
    real_rate = fred["dgs10"] - cpi_yoy
    macro = pd.DataFrame({
        "real_rate": real_rate,
        "d_real_rate_3m": real_rate.diff(3),
        "cpi_yoy": cpi_yoy,
        "unrate": fred["unrate"],
        "d_unrate_12m": fred["unrate"].diff(12),
        "vix": fred["vix"],
        "d_vix_3m": fred["vix"].diff(3),
        "dxy_ret_12m": fred["dxy"].pct_change(12),
    })
    cot_m = load_cot()
    macro["cot_net_pct_oi"] = cot_m
    macro["d_cot_3m"] = cot_m.diff(3)
    macro = macro.shift(1)  # 宏观与持仓特征统一滞后1个月，模拟发布延迟

    next_ret = r1.shift(-1)
    data = feat.join(macro)
    data["next_ret"] = next_ret
    data["label"] = (next_ret > 0).astype(int)
    FCOLS = [c for c in feat.columns] + [c for c in macro.columns]

    full = data.loc["1975-01-01":].dropna(subset=FCOLS)  # 特征完整即可
    train = full.dropna(subset=["next_ret"])             # 有标签的历史样本
    latest = full.iloc[[-1]]                           # 最新完整月 → 预测下一月
    last_month = full.index[-1]
    target_month = (last_month.to_period("M") + 1).to_timestamp()

    # 数据新鲜度检查
    now = pd.Timestamp.now()
    months_behind = (now.to_period("M") - last_month.to_period("M")).n
    stale = months_behind > 1

    sc = StandardScaler().fit(train[FCOLS])
    Xtr, ytr = sc.transform(train[FCOLS]), train["label"].values
    m1 = LogisticRegression(max_iter=3000, C=0.5).fit(Xtr, ytr)
    m2 = GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                    learning_rate=0.05, subsample=0.8, random_state=0).fit(Xtr, ytr)
    Xnew = sc.transform(latest[FCOLS])
    p1 = float(m1.predict_proba(Xnew)[0, 1])
    p2 = float(m2.predict_proba(Xnew)[0, 1])
    prob = (p1 + p2) / 2
    signal = "看多" if prob > 0.5 else "观望"
    confidence = abs(prob - 0.5) * 2  # 0~1

    row = full.iloc[-1]
    result = {
        "fetched_at": fetched_at,
        "last_complete_month": last_month.strftime("%Y-%m"),
        "target_month": target_month.strftime("%Y-%m"),
        "gold_close_last_month": round(float(px.loc[:last_month].iloc[-1]), 2),
        "prob_up": round(prob, 3),
        "prob_up_logistic": round(p1, 3),
        "prob_up_gbm": round(p2, 3),
        "signal": signal,
        "confidence": round(confidence, 3),
        "data_stale": bool(stale),
        "train_samples": int(len(train)),
        "macro_snapshot": {
            "real_rate": round(float(row["real_rate"]), 2),
            "d_real_rate_3m": round(float(row["d_real_rate_3m"]), 2),
            "cpi_yoy": round(float(row["cpi_yoy"]), 2),
            "unrate": round(float(row["unrate"]), 2),
            "vix": round(float(row["vix"]), 2),
            "dxy_ret_12m": round(float(row["dxy_ret_12m"]) * 100, 2),
            "gold_mom_12m": round(float(row["mom_12_1"]) * 100, 2),
        },
    }
    with open(f"{WS}/results/signal_latest.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n===== 黄金月度信号 =====")
    print(f"基于截至 {result['last_complete_month']} 的完整数据，预测 {result['target_month']} 月方向")
    print(f"信号：{signal}（上涨概率 {prob:.1%}，置信度 {confidence:.1%}）")
    if stale:
        print(f"⚠️ 金价历史停留在 {result['last_complete_month']}，距当前超过1个月，请先补充 gold_monthly_local.csv")
    print(f"宏观快照：实际利率 {result['macro_snapshot']['real_rate']}%，"
          f"CPI同比 {result['macro_snapshot']['cpi_yoy']}%，"
          f"VIX {result['macro_snapshot']['vix']}，"
          f"美元12月变动 {result['macro_snapshot']['dxy_ret_12m']}%，"
          f"黄金12月动量 {result['macro_snapshot']['gold_mom_12m']}%")
    print("结果已保存 signal_latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
