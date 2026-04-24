import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen2.5-7b-instruct-1m",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    stream=True
)
for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)


# {
#     "choices": [
#         {
#             "message": {
#                 "role": "assistant",
#                 "content": "我是阿里云开发的一款超大规模语言模型，我叫千问。"
#             },
#             "finish_reason": "stop",
#             "index": 0,
#             "logprobs": null
#         }
#     ],
#     "object": "chat.completion",
#     "usage": {
#         "prompt_tokens": 3019,
#         "completion_tokens": 104,
#         "total_tokens": 3123,
#         "prompt_tokens_details": {
#             "cached_tokens": 2048
#         }
#     },
#     "created": 1735120033,
#     "system_fingerprint": null,
#     "model": "qwen-plus",
#     "id": "chatcmpl-6ada9ed2-7f33-9de2-8bb0-78bd4035025a"
# }