from openai import OpenAI

# 1. 初始化設定 (維持你剛剛測試成功的 ws-02 設定)
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    api_key="vllm-token"
)

print("=== 開始對話 (輸入 'exit' 或 'q' 離開) ===")

# 2. 進入無限迴圈 (While Loop)
while True:
    # 接收使用者輸入
    user_input = input("User: ")
    
    # 設定離開條件
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break
    
    # 3. 發送請求 (API Call)
    try:
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",  # 這裡要用你伺服器支援的模型名稱
            messages=[
                # 設定 System Prompt (系統角色)，指示 AI 用繁體中文簡潔回答
                {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,   # 控制隨機性 (0.7 是通用設定)
            max_tokens=256     # 控制回答長度
        )
        
        # 4. 顯示 AI 回覆
        print(f"AI : {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"發生錯誤: {e}")