from openai import OpenAI

# 1. 初始化設定
# 使用投影片底部藍色區塊指定的 URL 和 Model
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1",  # 投影片下方指定的 URL
    api_key="vllm-token"                      # 根據投影片，API Key 可以填 vllm-token
)

print("正在發送請求...")

# 2. 發送對話請求
# 將原本的 prompt 放入 messages 結構中
response = client.chat.completions.create(
    model="google/gemma-3-27b-it",            # 投影片下方指定的模型名稱
    messages=[
        {"role": "user", "content": "你好，請自我介紹"}
    ],
    max_tokens=100                            # 對應你原本 code 的 max_new_tokens
)

# 3. 顯示結果
print("回應完成！")
print(response.choices[0].message.content)