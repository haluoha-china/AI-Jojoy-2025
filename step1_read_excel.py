import pandas as pd
from pathlib import Path

# 读取Excel文件
excel_path = Path('/root/autodl-tmp/enterprise_kb/sample_docs/公司常用缩略语20250401.xlsx')
df = pd.read_excel(excel_path)

# 建立缩写映射
abbr_map = {}
for _, row in df.iterrows():
    abbr = str(row.iloc[0]).strip().upper()
    eng = str(row.iloc[1]).strip()
    cn = str(row.iloc[2]).strip()
    abbr_map[abbr] = (eng, cn)

print(f"读取到 {len(abbr_map)} 个缩写")
print("前5个缩写:")
for i, (abbr, (eng, cn)) in enumerate(list(abbr_map.items())[:5]):
    print(f"{i+1}. {abbr} -> {cn} ({eng})")

# 保存映射到文件，供下一步使用
import pickle
with open('abbr_map.pkl', 'wb') as f:
    pickle.dump(abbr_map, f)
print("映射已保存到 abbr_map.pkl")
