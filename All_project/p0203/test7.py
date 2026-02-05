from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

# 1. 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

# 2. 定義兩個不同的 Prompt (提示詞樣板)
# 支線 A：負責寫詩
prompt_poem = ChatPromptTemplate.from_template("請寫一首關於 {topic} 的繁體中文短詩")
# 支線 B：負責講笑話
prompt_joke = ChatPromptTemplate.from_template("請講一個關於 {topic} 的繁體中文笑話")

# 3. 建立兩條 Chain (連鎖)
# 使用 LCEL 的魔術符號 "|" (Pipe)，意思就是：把 Prompt 的結果丟給 LLM
chain_a = prompt_poem | llm
chain_b = prompt_joke | llm

# 4. 定義平行處理 (Main Chain)
# 這會同時啟動 chain_a 和 chain_b
combined_chain = RunnableParallel(
    poem_result=chain_a, 
    joke_result=chain_b
)

print("=== 正在平行執行兩項任務 (寫詩 + 講笑話) ===")

# 5. 執行 (Invoke)
# 我們只需要傳入一次變數 "topic"，它會自動分發給兩條支線
final_output = combined_chain.invoke({"topic": "工程師"})

# 6. 顯示結果
print("\n[支線 A - 詩]:")
print(final_output["poem_result"].content)

print("\n" + "="*30 + "\n")

print("[支線 B - 笑話]:")
print(final_output["joke_result"].content)