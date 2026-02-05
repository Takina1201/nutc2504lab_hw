"""
LangChain 進階討論 - 實作(ch4-3)
科技文章摘要生成器 + 路由判斷

功能：
- 科技文章 → 使用 generate_tech_summary 工具生成摘要
- 閒聊/非科技文章 → 直接回覆

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# Import 所有必要的函式庫
# ============================================================
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. 設定 LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",  # KEY 留空
    model="google/gemma-3-27b-it",
    temperature=0
)


# ============================================================
# 2. 定義科技文章摘要工具
# ============================================================

@tool
def generate_tech_summary(article_content: str):
    """
    科技文章專用摘要生成工具。
    【判斷邏輯】：
    1. 只有當輸入內容屬於「科技」、「程式設計」、「AI」、「軟體工程」或「IT 技術」領域時，才使用此工具。
    2. 如果內容是「閒聊」、「食譜」、「天氣」、「日常日記」等非技術內容，請勿使用此工具。
    
    功能：將輸入的技術文章歸納出 3 個重點。
    """
    # 定義摘要專用的 Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個資深的科技主編。請將輸入的技術文章內容，精簡地歸納出 3 個關鍵重點 (Key Points)，並以繁體中文條列式輸出。"),
        ("user", "{text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"text": article_content})
    
    return result


# ============================================================
# 3. 註冊工具並建立路由 Chain
# ============================================================

llm_with_tools = llm.bind_tools([generate_tech_summary])

router_prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}")
])

chain = router_prompt | llm_with_tools


# ============================================================
# 4. 互動式對話迴圈
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 科技文章摘要助手 (ch4-3)")
    print("=" * 60)
    print("輸入科技文章 → 生成 3 個重點摘要")
    print("輸入閒聊 → 直接回覆")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        # 取得用戶輸入
        user_input = input("\nUser: ")
        
        # 檢查是否要離開
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break
        
        # 如果輸入為空，跳過
        if not user_input.strip():
            continue
        
        # 執行 chain
        ai_msg = chain.invoke({"input": user_input})
        
        # 判斷是否有工具呼叫
        if ai_msg.tool_calls:
            # ✅ 有工具呼叫 → 判斷為科技文章
            print(f"✅ [決策] 判斷為科技文章")
            
            # 取得工具參數
            tool_args = ai_msg.tool_calls[0]['args']
            
            # 執行工具（生成摘要）
            final_result = generate_tech_summary.invoke(tool_args)
            
            print(f"📄 [執行結果]:\n{final_result}")
        
        else:
            # ❌ 沒有工具呼叫 → 判斷為閒聊/非科技文章，直接回覆
            print(f"❌ [決策] 判斷為閒聊/非科技文章，直接回覆。")
            print(f"💬 [AI 回應]: {ai_msg.content}")