from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

# 2. 定義 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業的科技文章編輯。請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
    ("human", "{article_content}")
])

# 3. 定義 Parser
parser = StrOutputParser()

# 4. 建立 Chain
chain = prompt | llm | parser

# 5. 準備測試文章
tech_article = """
LangChain 是一個開源框架，旨在簡化使用大型語言模型 (LLM) 開發應用程式的過程。
它提供了一套工具和介面，讓開發者能夠將 LLM 與其他資料來源連接起來。
透過 LCEL 語法，開發者可以輕鬆地將不同的組件串聯在一起。
"""

print("=== 開始生成摘要 (串流模式) ===")

# 【關鍵修改】使用 .stream() 取代 .invoke()
# 這會回傳一個產生器 (Generator)，我們要用迴圈去接它
for chunk in chain.stream({"article_content": tech_article}):
    # end="" 代表不換行，讓字接在一起
    # flush=True 代表強制立即顯示，不要累積在緩衝區
    print(chunk, end="", flush=True)

print("\n") # 最後補一個換行，美觀