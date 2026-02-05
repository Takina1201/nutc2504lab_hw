from openai import OpenAI
import json

# 1. 初始化設定 (維持 ws-02)
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    api_key="vllm-token"
)

# 2. 模擬一段雜亂的客戶訂單文字 (這正是你所在的台中北區！)
user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

# 3. 設定 System Prompt (關鍵：明確要求 JSON 格式與欄位)
system_prompt = """你是一個資料提取助手。
請從使用者的文字中提取以下資訊，並嚴格以 JSON 格式回傳。
需要的欄位: name, phone, product, quantity, address"""

print(f"原始輸入: {user_input}\n" + "-"*30)

try:
    # 4. 發送請求
    response = client.chat.completions.create(
        model="google/gemma-3-27b-it",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        temperature=0.1,  # 【重要】設低溫度，讓 AI 不要亂加字，格式才會準
        max_tokens=256
    )
    
    raw_content = response.choices[0].message.content
    print(f"AI 原始回覆:\n{raw_content}\n" + "-"*30)

    # 5. 資料清洗 (重要步驟)
    # 有時候 AI 會雞婆加上 ```json ... ``` 的標記，這裡要把它去掉才能解析
    clean_json = raw_content.replace("```json", "").replace("```", "").strip()

    # 6. 解析並印出
    data = json.loads(clean_json) # 將字串轉為 Python 字典 (Dict)
    
    print("✅ 成功轉換為結構化資料:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    
except json.JSONDecodeError:
    print("❌ JSON 解析失敗：AI 回傳的格式不正確，可能包含多餘文字。")
except Exception as e:
    print(f"❌ 發生其他錯誤: {e}")