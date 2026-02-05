from openai import OpenAI
import sys

# 1. 初始化設定
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    api_key="vllm-token"
)

# 2. 初始化對話紀錄 (History)
# 這裡先放入系統提示詞，作為對話的開端
history = [
    {"role": "system", "content": "你是一個繁體中文的聊天機器人，請簡潔答覆"}
]

print("=== 開始多輪對話 (輸入 'exit' 或 'q' 離開) ===")

while True:
    user_input = input("\nUser: ")
    
    if user_input.lower() in ["exit", "q"]:
        print("Bye!")
        break
    
    # 【關鍵步驟 A】將使用者的話加入歷史紀錄
    history.append({"role": "user", "content": user_input})
    
    try:
        print("AI : (思考中...)", end="\r") # 做一個簡單的等待特效
        
        # 3. 發送請求 (帶入整個 history)
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",  
            messages=history,    # 注意：這裡傳送的是整個 history 列表
            temperature=0.7,
            max_tokens=256
        )
        
        full_reply = response.choices[0].message.content
        print(f"AI : {full_reply}")
        
        # 【關鍵步驟 B】將 AI 的回覆也加入歷史紀錄
        # 這樣下一輪對話時，AI 才會知道自己剛剛說了什麼
        history.append({"role": "assistant", "content": full_reply})
        
    except Exception as e:
        print(f"發生錯誤: {e}")