from src.components.qianfan_client import QianfanClient

client = QianfanClient(
    api_key="bce-v3/ALTAK-CXj6a7G9bgZLuyol7710b/10a27611f48c83c85723b42b066cae31462a921c",
    model="qwen2.5-7b-instruct",
)

resp = client.generate("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")
AGENT_IDS = ["A", "B", "C"]
print(resp)
import os
print("HTTP_PROXY =", os.environ.get("HTTP_PROXY"))
print("HTTPS_PROXY =", os.environ.get("HTTPS_PROXY"))
print("ALL_PROXY =", os.environ.get("ALL_PROXY"))