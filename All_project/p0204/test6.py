"""
簡易天氣查詢助理實作 - ch5-2
使用 LangGraph 建立天氣查詢 Agent

請先安裝: pip install langgraph langchain langchain-openai

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# 函式庫
# ============================================================
import json
from typing import Annotated, TypedDict, Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage  # 修正: message -> messages

# LangGraph 必要元件
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# LLM
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0  # 修正: temprature -> temperature
)


# ============================================================
# 工具 Tool
# ============================================================

@tool
def get_weather(city: str):
    """查詢指定城市的天氣。輸入參數 city 必須是城市名稱。"""
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
    messages: Annotated[list[BaseMessage], add_messages]  # 修正: message -> messages


# ============================================================
# 定義節點 (Nodes)
# ============================================================

def chatbot_node(state: AgentState):
    """思考節點：負責呼叫 LLM"""
    # 傳入目前對話紀錄，LLM 決定要回話還是呼叫工具
    response = llm_with_tools.invoke(state["messages"])  # 修正: message -> messages
    
    # 回傳的 dict 會自動合併進 State
    return {"messages": [response]}  # 修正: message -> messages


# 工具節點
tool_node_executor = ToolNode(tools)  # 修正: 使用 tools 變數


# ============================================================
# 定義邊 (Edges & Router)
# ============================================================

def router(state: AgentState) -> Literal["tools", "end"]:
    """路由邏輯：決定下一步是執行工具還是結束"""
    messages = state["messages"]  # 修正: message -> messages
    last_message = messages[-1]
    
    # 檢查最後一則訊息是否有 tool_calls
    if last_message.tool_calls:
        return "tools"
    else:
        return "end"


# ============================================================
# 組裝 Graph
# ============================================================

workflow = StateGraph(AgentState)

# (1) 加入節點
workflow.add_node("agent", chatbot_node)
workflow.add_node("tools", tool_node_executor)

# (2) 設定入口
workflow.set_entry_point("agent")

# (3) 設定條件邊 (Conditional Edge)
workflow.add_conditional_edges(
    "agent",        # 從 agent 出發
    router,         # 經過 router 判斷
    {
        "tools": "tools",  # 如果 router 回傳 "tools"，走向 tools 節點
        "end": END         # 如果 router 回傳 "end"，走向結束
    }
)

# (4) 循環邊
workflow.add_edge("tools", "agent")

# (5) 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("簡易天氣查詢助理 (ch5-2)")
    print("=" * 60)
    print("可查詢城市：台北、台中、高雄")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            print("Bye!")
            break
        
        if not user_input.strip():
            continue
        
        # 使用 stream 觀察執行過程
        for event in app.stream({"messages": [HumanMessage(content=user_input)]}):
            for key, value in event.items():
                last_msg = value["messages"][-1]
                
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"[AI 呼叫工具]: {last_msg.tool_calls}")
                elif hasattr(last_msg, 'content') and last_msg.content:
                    # 判斷是工具回傳還是 AI 回覆
                    if key == "tools":
                        print(f"[工具回傳]: {last_msg.content}")
                    else:
                        print(f"[AI]: {last_msg.content}")