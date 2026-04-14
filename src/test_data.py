import pandas as pd
import re
df = pd.read_parquet('datasets\\gsm8k\\main\\test-00000-of-00001.parquet')# 7473 1319

# 第一条
row = df.iloc[0]

print("问题：")
print(row['question'])
print("\n答案：")
result = re.findall(r'####\s*(\d+)', row['answer'])
print(result[0] if result else None)

print(len(df))
