import requests

url = "http://localhost:11434/api/chat"
model = "deepseek-r1:7b"
headers = {"Content-Type": "application/json"}
data = {
    "model": model,  # 模型选择
    "options": {
        "temperature": 0.1  # 为0表示不让模型自由发挥，输出结果相对较固定，>0的话，输出的结果会比较放飞自我
    },
    "stream": False,  # 流式输出
    "messages": [{"role": "system", "content": "你是谁？"}],  # 对话列表
}
response = requests.post(url, json=data, headers=headers, timeout=600)
res = response.json()
print(res["message"]["content"])  # 输出对话结果
