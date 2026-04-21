import pandas as pd
import re

# df = pd.read_parquet('datasets\\gsm8k\\main\\train-00000-of-00001.parquet')

# # 统计变量
# total_count = len(df)
# integer_answer_count = 0
# non_integer_answers = []

# # 遍历每一条数据
# for idx, row in df.iterrows():
#     answer_text = row['answer']
    
#     # 提取 #### 后面的数字
#     result = re.findall(r'####\s*(-?\d+)', answer_text)
    
#     if result:
#         ans = result[0]
#         # 判断是不是纯整数（GSM8K 答案都是正整数）
#         num = float(ans)
#         if num.is_integer():
#             integer_answer_count += 1
#         else:
#             non_integer_answers.append((idx, ans))
#     else:
#         # 没匹配到答案
#         non_integer_answers.append((idx, "无答案"))
#         print(answer_text)

# # 输出结果
# print("="*50)
# print(f"总数据条数：{total_count}")
# print(f"答案是整数的条数：{integer_answer_count}")
# print(f"答案不是整数的条数：{len(non_integer_answers)}")
# print("="*50)

# # 如果有非整数答案，打印出来查看
# if non_integer_answers:
#     print("\n以下是答案不是整数的条目（索引 | 答案）：")
#     for idx, ans in non_integer_answers:
#         print(f"第 {idx} 条：{ans}")
# else:
#     print("\n✅ 所有答案都是整数！")




# 读取 parquet 文件
df = pd.read_parquet('datasets\\aime_2026\\data\\train-00000-of-00001.parquet')

# ===================== 基础查看 =====================
print("===== 数据形状（行数, 列数）=====")
print(df.shape)

print("\n===== 列名 =====")
print(df.columns.tolist())

print("\n===== 数据类型 =====")
print(df.dtypes)

print("\n===== 前 5 行数据 =====")
print(df.head())

print("\n===== 前 3 行详细展示（每条完整内容） =====")
for i in range(3):
    print(f"\n========== 第 {i} 条数据 ==========")
    print(df.iloc[i])

# ===================== 单独查看字段 =====================
print("\n===== 查看所有题目索引 =====")
print(df['problem_idx'].tolist())

print("\n===== 查看所有答案 =====")
print(df['answer'].tolist())

print("\n===== 查看第一条完整题目 =====")
print(df['problem'].iloc[0])