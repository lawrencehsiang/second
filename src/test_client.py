import os

from dotenv import load_dotenv

from src.components.qianfan_client import QianfanClient
# 强制禁用代理，直连国内网络
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "qianfan.baidubce.com,localhost,127.0.0.1"
load_dotenv()
api_key = os.getenv("QIANFAN_API_KEY")
if not api_key:
    raise ValueError("Missing QIANFAN_API_KEY. Please set it in your .env file.")

client = QianfanClient(
    api_key=api_key,
    model="qwen2.5-7b-instruct",
)

resp = client.generate_with_usage("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
AGENT_IDS = ["A", "B", "C"]
print(resp)
raw_text = resp["content"]
usage = resp["usage"]
print(raw_text)
print(usage)
print("HTTP_PROXY =", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY =", os.environ.get("HTTPS_PROXY"))
print("ALL_PROXY =", os.environ.get("ALL_PROXY"))
