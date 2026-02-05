"""
LangGraph 進階應用 - 實作(ch6-3)
人工審核的訂單資訊 (Human-in-the-Loop)

特點：
1. 定義 VIP 名單，VIP 客戶的訂單需要人工審核
2. 普通客戶的訂單直接處理
3. 人工審核時，流程會暫停等待管理員輸入

流程圖（加入 human_review 後）：
    __start__
        |
        v
     agent  <-----------+
        |               |
   [entry_router]       |
    /        \\         |
   v          v        |
 END        tools      |
              |        |
       [post_tool_router]
         /        \\    |
        v          v   |
    agent    human_review
              |        |
              +--------+

請先安裝: pip install langgraph langchain langchain-openai

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# 函式庫
# ============================================================
import json
from typing import Annotated, TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 配置區
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0
)

# 定義 VIP 名單（這就是觸發人工審核的關鍵）
VIP_LIST = ["AI哥", "一龍馬"]


# ============================================================
# 工具定義
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
# 綁定工具
# ============================================================

llm_with_tools = llm.bind_tools([extract_order_data])
tool_node = ToolNode([extract_order_data])


# ============================================================
# State 定義
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 節點定義
# ============================================================

def agent_node(state: AgentState):
    """思考節點"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def human_review_node(state: AgentState):
    """
    [新增] 人工審核節點
    當檢測到 VIP 時，流程會卡在這裡等待人工輸入。
    """
    print("\n" + "=" * 30)
    print("觸發人工審核機制：檢測到 VIP 客戶！")
    print("=" * 30)
    
    # 取得上一則 ToolMessage 的內容來顯示
    last_msg = state["messages"][-1]
    print(f"待審核資料: {last_msg.content}")
    
    # 模擬人工決策（在真實系統中，這裡可能會發送 Slack 通知或暫停 Graph）
    review = input(">>> 管理員請批示（輸入 'ok' 通過，其他則拒絕）: ")
    
    if review.lower() == "ok":
        return {
            "messages": [
                AIMessage(content="已收到訂單資料，因偵測到 VIP 客戶，系統將轉交人工審核..."),
                HumanMessage(content="[系統公告] 管理員已人工審核通過此 VIP 訂單，請繼續完成後續動作。")
            ]
        }
    else:
        return {
            "messages": [
                AIMessage(content="已收到訂單資料，等待人工審核結果..."),
                HumanMessage(content="[系統公告] 管理員拒絕了此訂單，請取消交易並告知用戶。")
            ]
        }


# ============================================================
# 路由定義
# ============================================================

def entry_router(state: AgentState):
    """判斷 Agent 是否要呼叫工具"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


def post_tool_router(state: AgentState) -> Literal["human_review", "agent"]:
    """
    [新增] 工具執行後的路由邏輯
    檢查工具回傳的內容，決定要給 AI 繼續處理，還是轉給人工。
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 確保是工具回傳的訊息
    if isinstance(last_message, ToolMessage):
        try:
            # 解析工具回傳的 JSON 字串
            data = json.loads(last_message.content)
            user_name = data.get("name", "")
            
            # === 核心判斷邏輯 ===
            if user_name in VIP_LIST:
                print(f"DEBUG: 發現 VIP [{user_name}] -> 轉向人工審核")
                return "human_review"
        except Exception as e:
            print(f"JSON 解析錯誤: {e}")
    
    # 如果不是 VIP 或解析失敗，就走正常流程回到 Agent
    return "agent"


# ============================================================
# 組裝 Graph
# ============================================================

workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("human_review", human_review_node)  # 加入人工節點

# 設定入口
workflow.set_entry_point("agent")

# (1) Agent -> 工具 或 結束
workflow.add_conditional_edges(
    "agent",
    entry_router,
    {"tools": "tools", END: END}
)

# (2) [修改] 工具 -> 判斷是 VIP 還是 普通人
workflow.add_conditional_edges(
    "tools",
    post_tool_router,
    {
        "human_review": "human_review",  # 走人工
        "agent": "agent"                  # 走回 AI (Loop)
    }
)

# (3) 人工審核完 -> 回到 Agent 讓他做總結
workflow.add_edge("human_review", "agent")

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print(f"\nVIP 名單: {VIP_LIST}")
    print("\n" + "=" * 60)
    print("LangGraph 進階應用 - 人工審核訂單系統 (ch6-3)")
    print("=" * 60)
    print("VIP 客戶的訂單會觸發人工審核")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            break
        
        if not user_input.strip():
            continue
        
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                if key == "agent":
                    # 顯示 AI 的回覆
                    msg = value["messages"][-1]
                    if not msg.tool_calls:
                        print(f"-> [Agent]: {msg.content}")
                elif key == "human_review":
                    # 顯示人工審核的結果
                    print(f"-> [Human]: 審核完成")