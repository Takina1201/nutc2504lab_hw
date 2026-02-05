"""
LangGraph 進階應用 - 實作(ch6-1)
Retry 機制的天氣 API (ch5-2 的主題延伸)

特點：
1. 天氣 API 會隨機失敗（模擬真實情況）
2. 加入 Retry 機制，失敗時自動重試
3. 設定重試上限（3次），超過則觸發 fallback 節點

流程圖：
    __start__
        |
        v
     agent  <-----------+
        |               |
   [router 判斷]        |
    /    |    \\        |
   v     v     v       |
 END  tools  fallback  |
        |       |      |
        +-------+------+

請先安裝: pip install langgraph langchain langchain-openai

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# 函式庫
# ============================================================
import random
import json
import os
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
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


# ============================================================
# 工具定義（故意讓他出錯）
# ============================================================

@tool
def get_weather(city: str):
    """查詢指定城市的天氣。"""
    # 50% 機率失敗（模擬不穩定的 API）
    if random.random() < 0.5:
        return "系統錯誤：天氣資料庫連線失敗，請再試一次。"
    
    if "台北" in city:
        return "台北下大雨，氣溫 18 度"
    elif "台中" in city:
        return "台中晴天，氣溫 26 度"
    elif "高雄" in city:
        return "高雄多雲，氣溫 30 度"
    else:
        return "資料庫沒有這個城市的資料"


# ============================================================
# 綁定工具
# ============================================================

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# State 定義
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 節點定義
# ============================================================

def chatbot_node(state: AgentState):
    """思考節點"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node_executor = ToolNode(tools)


def fallback_node(state: AgentState):
    """
    [新增] 備援節點：當重試次數過多時執行。
    它必須回傳一個 ToolMessage，這樣 LLM 才會覺得工具流程已經結束了。
    """
    last_message = state["messages"][-1]
    
    # 抓取上一個 AI 想要呼叫的 tool_call_id，偽造一個失敗的回傳
    tool_call_id = last_message.tool_calls[0]["id"]
    
    error_message = ToolMessage(
        content="系統警示：已達到最大重試次數 (Max Retries Reached)。請停止嘗試，並告知使用者服務暫時無法使用。",
        tool_call_id=tool_call_id
    )
    return {"messages": [error_message]}


# ============================================================
# 定義邊 (Edges & Router)
# ============================================================

def router(state: AgentState) -> Literal["tools", "fallback", "end"]:
    """
    [修改] 路由邏輯：
    1. 檢查是否有 tool_calls
    2. [新增] 檢查是否已經連續失敗太多次
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # 如果沒有 tool_calls，結束
    if not last_message.tool_calls:
        return "end"
    
    # --- 計算連續錯誤次數 ---
    # 我們往回看歷史訊息，計算有多少個連續的「系統錯誤」
    retry_count = 0
    
    # 從倒數第 2 則開始往回找（因為倒數第 1 則是剛產生的 AI ToolCall）
    for msg in reversed(messages[:-1]):
        if isinstance(msg, ToolMessage):
            if "系統錯誤" in msg.content:
                retry_count += 1
            else:
                # 如果遇到一次成功的，或不是錯誤的，就中斷計算
                break
        elif isinstance(msg, HumanMessage):
            # 如果回溯到使用者發言，就停止
            break
    
    print(f"DEBUG: 目前連續重試次數: {retry_count}")
    
    # 設定上限為 3 次
    if retry_count >= 3:
        return "fallback"  # 導向備援節點
    
    return "tools"  # 正常執行工具


# ============================================================
# 組裝 Graph
# ============================================================

workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)
workflow.add_node("fallback", fallback_node)  # 加入 fallback 節點

# 設定入口
workflow.set_entry_point("agent")

# 設定條件邊
workflow.add_conditional_edges(
    "agent",
    router,
    {
        "tools": "tools",
        "fallback": "fallback",  # 路線 B：超過上限，去 fallback
        "end": END
    }
)

# 循環邊
workflow.add_edge("tools", "agent")
workflow.add_edge("fallback", "agent")  # fallback 執行完也要回給 agent，讓它做最後結論

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangGraph 進階應用 - Retry 機制天氣查詢 (ch6-1)")
    print("=" * 60)
    print("天氣 API 有 50% 機率失敗，會自動重試")
    print("重試超過 3 次會觸發 fallback 機制")
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
                    msg = value["messages"][-1]
                    if msg.tool_calls:
                        print(f" -> [Agent]: 決定呼叫工具（重試中...）")
                    else:
                        print(f" -> [Agent]: {msg.content}")
                elif key == "tools":
                    msg = value["messages"][-1]
                    if "系統錯誤" in msg.content:
                        print(f" -> [Tools]: 系統錯誤...（失敗）")
                    else:
                        print(f" -> [Tools]: {msg.content}")
                elif key == "fallback":
                    print(f" -> [Fallback]: 觸發熔斷機制：停止重試")