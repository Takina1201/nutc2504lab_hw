from langchain_openai import ChatOpenAI

# 1. 設定 LLM (對接 vLLM)
# 這裡我們用 ChatOpenAI 這個通用介面，來連線你的遠端模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",  # 維持你測試成功的網址
    api_key="vllm-token",
    model="google/gemma-3-27b-it",             # 維持你測試成功的模型
    temperature=0.7,
    max_tokens=256
)

print("=== 正在測試 LangChain 標準化介面 ===")

# 2. 發送請求 (Invoke)
# 注意：在 LangChain 裡，我們不再用 .create()，而是統一用 .invoke()
try:
    response = llm.invoke("你好，請用一句話解釋為什麼『標準化』對寫程式很重要？")
    
    # 3. 顯示結果
    # LangChain 回傳的是一個 AIMessage 物件，內容在 .content 屬性裡
    print(f"AI 回答: {response.content}")

except Exception as e:
    print(f"發生錯誤: {e}")