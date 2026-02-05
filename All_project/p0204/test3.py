"""
LangChain 進階討論 - 實作(ch4-2)
ch4-1 的改進版：加入互動式對話迴圈

改進重點：
- 沒調用工具時，直接輸出非結構化內容（ai_message.content）
- 加入 while True 互動迴圈，可持續對話
- 輸入 "exit" 或 "q" 離開

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# Import 所有必要的函式庫
# ============================================================
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json


# ============================================================
# 1. 定義工具
# ============================================================

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }


# ============================================================
# 2. 設定 LLM 並註冊工具
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",  # KEY 留空
    model="google/gemma-3-27b-it",
    temperature=0
)

llm_with_tools = llm.bind_tools([extract_order_data])


# ============================================================
# 3. 建立 Prompt Template
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])


# ============================================================
# 4. 改進版的提取函數（重點！）
# ============================================================

def extract_tool_args(ai_message):
    """
    從 AI 回應中提取工具呼叫的參數
    
    改進：如果沒有調用工具，就直接返回 AI 的回覆內容
    """
    if ai_message.tool_calls:
        # 有工具呼叫 → 返回工具參數（結構化資料）
        return ai_message.tool_calls[0]['args']
    else:
        # 沒有工具呼叫 → 返回 AI 的直接回覆（非結構化內容）
        return ai_message.content


# ============================================================
# 5. 建立 Chain
# ============================================================

chain = prompt | llm_with_tools | extract_tool_args


# ============================================================
# 6. 互動式對話迴圈
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 訂單管理助手 (ch4-2 改進版)")
    print("=" * 60)
    print("輸入訂單資訊，我會幫你提取結構化資料。")
    print("輸入 'exit' 或 'q' 離開。")
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
        result = chain.invoke({"user_input": user_input})
        
        # 輸出結果
        print(json.dumps(result, ensure_ascii=False, indent=2))