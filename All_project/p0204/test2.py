"""
LangChain 進階討論 - 實作(ch4-1)
使用 LCEL (LangChain Expression Language) 的 chain 寫法

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
# 1. 定義工具 (Tool)
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
# 2. 設定 LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",  # KEY 留空
    model="google/gemma-3-27b-it",
    temperature=0
)


# ============================================================
# 3. 註冊工具
# ============================================================

llm_with_tools = llm.bind_tools([extract_order_data])


# ============================================================
# 4. 建立 Prompt Template
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一個精準的訂單管理員，請從對話中提取訂單資訊。"),
    ("user", "{user_input}")
])


# ============================================================
# 5. 定義提取工具參數的函數
# ============================================================

def extract_tool_args(ai_message):
    """從 AI 回應中提取工具呼叫的參數"""
    if ai_message.tool_calls:
        return ai_message.tool_calls[0]['args']
    return None


# ============================================================
# 6. 建立 Chain (使用 LCEL 的 pipe 語法)
# ============================================================

chain = prompt | llm_with_tools | extract_tool_args


# ============================================================
# 7. 執行測試
# ============================================================

if __name__ == "__main__":
    # 測試用的用戶輸入
    user_text = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"
    
    print("=" * 60)
    print("📝 用戶輸入:")
    print(user_text)
    print("=" * 60)
    
    # 執行 chain
    result = chain.invoke({"user_input": user_text})
    
    # 顯示結果
    if result:
        print("✅ 提取成功:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("❌ 提取失敗")