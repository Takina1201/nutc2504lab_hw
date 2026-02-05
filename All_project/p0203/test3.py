from openai import OpenAI

# 1. 初始化設定
client = OpenAI(
    base_url="https://ws-02.wade0426.me/v1", 
    api_key="vllm-token"
)

# 2. 設定測試用的 Prompt 和不同的溫度值
prompt = "請用100字形容『人工智慧』。"
temps = [0.1, 1.5]  # 0.1 很冷靜 (固定), 1.5 很發散 (創意/混亂)

print(f"提示詞: {prompt}\n")

# 3. 迴圈測試不同溫度
for t in temps:
    print(f"➡ 測試 Temperature = {t} ...")
    try:
        response = client.chat.completions.create(
            model="google/gemma-3-27b-it",  # 使用目前可用的模型
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=t,      # 這裡帶入迴圈目前的溫度
            max_tokens=200      # 稍微加長一點讓它發揮
        )
        
        # 顯示結果
        print(f"🤖 回覆: {response.choices[0].message.content}\n" + "-"*30)
        
    except Exception as e:
        print(f"❌ 發生錯誤: {e}")