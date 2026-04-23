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


# df = pd.read_parquet('datasets\\SVAMP\\data\\train-00000-of-00001.parquet')

# # ===================== 基础查看 ====================


# output_path = "svamp.jsonl"

# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
        
#         item = {
#             "question": row["Body"]+row["Question"],
#             "answer": row["Answer"]
#         }
        
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("✅ 转换完成！无 ndarray 报错")

# with open("datasets\\multiarith\\MultiArith.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 后面逻辑完全一样
# output_path = "multiarith.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in data:
#         question = item["sQuestion"].strip()
#         answer = item["lSolutions"][0]
#         line = {"question": question, "answer": answer}
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")

# print("✅ 转换完成！")


# with open("datasets\\singleeq\\SingleEq.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 后面逻辑完全一样
# output_path = "singleeq.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in data:
#         question = item["sQuestion"].strip()
#         answer = item["lSolutions"][0]
#         line = {"question": question, "answer": answer}
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")

# print("✅ 转换完成！")


# with open("datasets\\addsub\\AddSub.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# # 后面逻辑完全一样
# output_path = "addsub.jsonl"
# with open(output_path, "w", encoding="utf-8") as f:
#     for item in data:
#         question = item["sQuestion"].strip()
#         answer = item["lSolutions"][0]
#         line = {"question": question, "answer": answer}
#         f.write(json.dumps(line, ensure_ascii=False) + "\n")

# print("✅ 转换完成！")

# df = pd.read_parquet('datasets\\asdiv\\asdiv\\asdiv\\validation-00000-of-00001.parquet')


# output_path = "asdiv.jsonl"

# def extract_number(text):
#     # 匹配所有数字（支持整数、小数）
#     numbers = re.findall(r'\d+\.?\d*', str(text))
#     if numbers:
#         return float(numbers[0])  # 返回数字类型
#     return None  # 没找到数字返回空

# with open(output_path, "w", encoding="utf-8") as f:
#     for _, row in df.iterrows():
        
#         item = {
#             "question": row["body"]+row["question"],
#             "answer": extract_number(row["answer"])
#         }
        
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("✅ 转换完成！无 ndarray 报错")

df = pd.read_parquet('datasets\\math\\competition_math\\data\\train-00000-of-00001-7320a6f3aba8ebd2.parquet')

# 2. 定义函数：提取 \boxed{} 里的答案
def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{(.*?)\}', str(text))
    if match:
        return match.group(1).strip()
    return str(text).strip()

# 3. 定义函数：判断并转换为合法整数（过滤掉非整数、分数、小数）
def is_integer(ans):
    try:
        # 尝试转成整数
        num = int(ans)
        return True, num  # 是整数，返回数值
    except (ValueError, TypeError):
        return False, None  # 不是整数

# 4. 处理并保存【只含整数答案】的数据
output_path = "math_integer_only.jsonl"

with open(output_path, "w", encoding="utf-8") as f:
    for item in df.to_dict("records"):
        question = item["problem"].strip()
        solution = item["solution"]
        
        # 清洗答案
        clean_answer = extract_boxed_answer(solution)
        
        # 判断是否为整数
        is_int, int_ans = is_integer(clean_answer)
        
        # ✅ 只保留整数答案
        if is_int:
            line = {
                "question": question,
                "answer": int_ans  # 存为整数类型
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

print("✅ 完成！已生成只包含整数答案的文件：math_integer_only.jsonl")