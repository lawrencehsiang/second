import pandas as pd
import re
import json
import numpy as np
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
# df = pd.read_parquet('datasets\\mmlu\\all\\dev-00000-of-00001.parquet')

# # ===================== 基础查看 ====================


# output_path = "mmlu_dev.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
#         # 关键修复：把 numpy array 转成普通 list
#         choices = row["choices"].tolist() if isinstance(row["choices"], np.ndarray) else row["choices"]
        
#         item = {
#             "question": row["question"],
#             "choices": choices,  # 现在是列表，可序列化
#             "answer": int(row["answer"])
#         }
        
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("✅ 转换完成！无 ndarray 报错")


# 读取两个数据集
# df_test = pd.read_parquet('datasets\\MMLU-Pro\\data\\test-00000-of-00001.parquet')
# df_val = pd.read_parquet('datasets\\MMLU-Pro\\data\\validation-00000-of-00001.parquet')

# # 合并成一个 DataFrame
# df_combined = pd.concat([df_test, df_val], ignore_index=True)

# # 写入同一个 JSONL 文件
# with open("mmlu_pro_combined.jsonl", "w", encoding="utf-8") as f:
#     for _, row in df_combined.iterrows():
#         # numpy array 转列表（防报错）
#         choices = row["options"].tolist() if isinstance(row["options"], np.ndarray) else row["options"]
        
#         # 你要的最终格式！
#         item = {
#             "question": row["question"],
#             "choices": choices,
#             "answer": int(row["answer_index"])
#         }
        
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("✅ 合并完成！文件：mmlu_pro_combined.jsonl")


df = pd.read_parquet('datasets\\SVAMP\\data\\train-00000-of-00001.parquet')

# ===================== 基础查看 ====================


output_path = "svamp.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        
        item = {
            "question": row["Body"]+row["Question"],
            "answer": row["Answer"]
        }
        
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ 转换完成！无 ndarray 报错")