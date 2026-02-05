"""
LangGraph 核心元件拆解 - 實作(ch5-1)
取自 ch4-1，使用 LangGraph 重構

請先安裝: pip install langgraph

核心元件：
1. State (狀態) - Agent 的記憶體
2. Nodes (節點) - 執行邏輯的單元
3. Edges (邊與決策) - 節點之間的連接和條件判斷

流程圖：
    __start__
        |
        v
     agent  <----+
        |        |
   [決策判斷]    |
    /     \\     |
   v       v    |
 END    tools --+

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# Import 所有必要的函式庫
# ============================================================
import json
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage

# LangGraph 必要元件
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 1. 設定 LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)


# ============================================================
# 2. 定義工具
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


# 綁定工具
llm_with_tools = llm.bind_tools([extract_order_data])


# ============================================================
# 3. 元件一：State（狀態）
# ============================================================
# 這是 Agent 的記憶體，add_messages 確保訊息是「疊加」而非覆蓋

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 會自動保留對話紀錄


# ============================================================
# 4. 元件二：Nodes（節點）
# ============================================================

# Node A: 思考節點（負責呼叫 LLM）
def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Node B: 工具節點（LangGraph 內建的 ToolNode，省去自己寫執行邏輯）
tool_node = ToolNode([extract_order_data])


# ============================================================
# 5. 元件三：Edges（邊與決策）
# ============================================================

# 判斷邏輯：LLM 有呼叫工具 -> 走工具節點；沒有 -> 結束
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# ============================================================
# 6. 組裝 Graph
# ============================================================

# 初始化 Graph，並給賦予為 AgentState 的 state
workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("agent", call_model)    # 加入節點 A
workflow.add_node("tools", tool_node)     # 加入節點 B

# 設定起點
workflow.set_entry_point("agent")

# 設定條件邊 (Conditional Edge)
# 從節點 A 做完事之後...
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}  # LLM 有呼叫工具 -> 走工具節點；沒有 -> 結束
)

# 設定循環邊 (Loop): 工具做完後，把結果丟回給 Agent 再看一次
workflow.add_edge("tools", "agent")


# ============================================================
# 7. 編譯並執行
# ============================================================

app = workflow.compile()

# 印出流程圖（ASCII 版本）
print(app.get_graph().draw_ascii())


# ============================================================
# 8. 測試執行
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangGraph Agent (ch5-1)")
    print("=" * 60)
    print("輸入訂單資訊或閒聊，觀察 Agent 的執行流程。")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            break
        
        # 使用 stream 看過程，或 invoke 直接拿結果
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                # 這裡簡單印出最後一條訊息內容觀測
                last_msg = value["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(last_msg.tool_calls)
                elif hasattr(last_msg, 'content'):
                    print(last_msg.content)
                else:
                    print(last_msg)