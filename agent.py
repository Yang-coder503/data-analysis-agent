import os
from dotenv import load_dotenv
import anthropic

# 职责
# 1. 加载系统提示词
# 2. 识别任务类型
# 3. 加载对应task_profile
# 4. 按phase处理数据
# 5. 管理对话历史

load_dotenv()  # 加载 .env 文件

ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

client = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    base_url=ANTHROPIC_BASE_URL
)

history = []

while True:
    user_input = input("需要我做些什么？").strip()

    history.append({"role": "user", "content": user_input})

    message = client.messages.create(
            model='deepseek-v4-pro',
            max_tokens=5000,
            messages=[{"role": "user", "content": user_input}]
    )   

    reply = next(b.text for b in message.content if b.type == "text")
    print(f"AI: {reply}\n")
    history.append({"role": "assistant", "content": reply})
