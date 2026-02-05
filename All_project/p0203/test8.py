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

# 2. 定義 Prompt Template (提示詞樣板)
# 使用 from_messages 可以更清楚分開 "系統設定" 與 "使用者輸入"
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個專業的科技文章編輯。請將使用者提供的文章內容，歸納出 3 個重點，並以繁體中文條列式輸出。"),
    ("human", "{article_content}")
])

# 3. 定義 Output Parser (輸出解析器)
# 它的作用是直接把 AI 的 Message 物件轉成純字串，省去寫 .content 的麻煩
parser = StrOutputParser()

# 4. 建立 Chain (黃金組合：提示詞 -> 模型 -> 解析器)
chain = prompt | llm | parser

# 5. 準備測試文章 (關於 LangChain 的介紹)
tech_article = """
LangChain 是一個開源框架，旨在簡化使用大型語言模型 (LLM) 開發應用程式的過程。
它提供了一套工具和介面，讓開發者能夠將 LLM 與其他資料來源 (如網際網路或個人檔案) 連接起來。
LangChain 的核心概念包括 Chain (鏈)、Agent (代理) 和 Memory (記憶)。
透過 LCEL 語法，開發者可以輕鬆地將不同的組件串聯在一起，構建複雜的 AI 應用。
最近，LangChain 推出了 LangGraph，專門用於構建具備循環邏輯的有狀態代理。
"""

print("=== 開始生成摘要 ===")

# 6. 執行 (Invoke)
# 因為加了 parser，這裡回傳的 result 直接就是字串了！
result = chain.invoke({"article_content": tech_article})

print(result)