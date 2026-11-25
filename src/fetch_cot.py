# -*- coding: utf-8 -*-
"""下载 CFTC 年度 COT 文件（2006-2026），提取 COMEX 黄金非商业持仓，输出月度特征"""
import io
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path("/Users/mac/Documents/kimi/tasks/2026-11-24/14-52-56-7456557d")
OUT = WS / "cot_gold_weekly.csv"

# deacot 年度文件为固定宽度逗号混合文本；列位置（legacy futures-only）：
# 0 Market_and_Exchange_Names, 1 As_of_Date(mmddyy), 2 Report_Date(yyyy-mm-dd),
# 3 CFTC_Contract_Market_Code, 7 Open_Interest_All,
# 8 NonComm_Long_All, 9 NonComm_Short_All, 11 Comm_Long_All, 12 Comm_Short_All
rows = []
for year in range(2006, 2027):
    url = f"https://www.cftc.gov/files/dea/history/deacot{year}.zip"
    try:
        raw = subprocess.run(["curl", "-sL", "--max-time", "90", url],
                             capture_output=True, timeout=100).stdout
        z = zipfile.ZipFile(io.BytesIO(raw))
        name = z.namelist()[0]
        df = pd.read_csv(z.open(name), header=0, low_memory=False)
        gold = df[df["Market and Exchange Names"].str.strip() == "GOLD - COMMODITY EXCHANGE INC."]
        cols = {"As of Date in Form YYYY-MM-DD": "date",
                "Open Interest (All)": "oi",
                "Noncommercial Positions-Long (All)": "nc_long",
                "Noncommercial Positions-Short (All)": "nc_short",
                "Commercial Positions-Long (All)": "c_long",
                "Commercial Positions-Short (All)": "c_short"}
        gold = gold.rename(columns=cols)[list(cols.values())]
        rows.append(gold)
        print(f"{year}: {len(gold)} 周")
    except Exception as e:  # noqa: BLE001
        print(f"{year}: 失败 {e}")

cot = pd.concat(rows).drop_duplicates("date")
cot["date"] = pd.to_datetime(cot["date"])
cot = cot.set_index("date").sort_index()
for c in ("oi", "nc_long", "nc_short", "c_long", "c_short"):
    cot[c] = pd.to_numeric(cot[c], errors="coerce")
cot["nc_net"] = cot["nc_long"] - cot["nc_short"]
cot["nc_net_pct_oi"] = cot["nc_net"] / cot["oi"]
print(f"合计 {len(cot)} 周，范围 {cot.index[0].date()} .. {cot.index[-1].date()}")
print(cot.tail(3))
cot.to_csv(OUT)
print(f"saved {OUT}")
