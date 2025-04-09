import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-1c5c4fb2b35646b5bba7491c1acd05ff",  # 替换为你的通义千问API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-vl-plus",  # 此处以qwen-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=[{"role": "user","content": [
            {"type": "text","text": "这是什么"},
            {"type": "image_url",
             "image_url": {"url": "https://9525-36-161-107-179.ngrok-free.app/static/uploads/dog_and_girl.jpeg"}}
            ]}]
    )
print(completion.model_dump_json())