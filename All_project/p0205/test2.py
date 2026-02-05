"""
LangGraph 進階應用 - 實作(ch6-2)
反饋機制的翻譯機 (Reflection Pattern)

特點：
1. translator 節點：負責翻譯中文到英文
2. reflector 節點：負責審查翻譯品質
3. 如果審查不通過，會退回重寫
4. 設定重試上限（3次），防止無窮迴圈

流程圖：
    __start__
        |
        v
    translator  <---+
        |           |
        v           |
    reflector ------+
        |
        v (PASS 或達到上限)
     __end__

請先安裝: pip install langgraph langchain langchain-openai

API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

# ============================================================
# 函式庫
# ============================================================
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


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
# 1. 定義狀態
# ============================================================

class State(TypedDict):
    original_text: str      # 原始文字
    translated_text: str    # 翻譯結果
    critique: str           # 評語
    attempts: int           # 重試次數（防止無窮迴圈）


# ============================================================
# 2. 定義節點
# ============================================================

def translator_node(state: State):
    """負責翻譯的節點"""
    print(f"\n--- 翻譯嘗試（第 {state['attempts'] + 1} 次）---")
    
    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋: '{state['original_text']}'"
    
    # 如果有上一輪的審查意見，加入 prompt
    if state['critique']:
        prompt += f"\n\n上一輪的審查意見是: {state['critique']}。請根據意見修正翻譯。"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "translated_text": response.content,
        "attempts": state['attempts'] + 1
    }


def reflector_node(state: State):
    """負責審查的節點 (Critique)"""
    print("--- 審查中 (Reflection) ---")
    print(f"翻譯: {state['translated_text']}")
    
    prompt = f"""
你是一個嚴格的翻譯審查員。

原文: {state['original_text']}
翻譯: {state['translated_text']}

請檢查翻譯是否準確且通順。
- 如果翻譯很完美，請只回覆 "PASS"。
- 如果需要修改，請給出簡短的具體建議。
"""
    
    # 呼叫 LLM
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"critique": response.content}


# ============================================================
# 3. 定義邊 (Edge) - 決策邏輯
# ============================================================

def should_continue(state: State) -> Literal["translator", "end"]:
    critique = state['critique'].strip().upper()
    
    if "PASS" in critique:
        print("--- 審查通過！---")
        return "end"
    elif state['attempts'] >= 3:
        print("--- 達到最大重試次數，強制結束 ---")
        return "end"
    else:
        print(f"--- 審查未通過: {state['critique']} ---")
        print("--- 退回重寫 ---")
        return "translator"


# ============================================================
# 4. 組裝 Graph
# ============================================================

workflow = StateGraph(State)

# 加入節點
workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

# 設定入口
workflow.set_entry_point("translator")

# 設定邊
workflow.add_edge("translator", "reflector")

# 設定條件邊
workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "translator": "translator",
        "end": END
    }
)

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LangGraph 進階應用 - 反饋機制翻譯機 (ch6-2)")
    print("=" * 60)
    print("輸入中文，會自動翻譯成英文")
    print("翻譯後會經過審查，不通過會自動修正")
    print("最多重試 3 次")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "q"]:
            break
        
        if not user_input.strip():
            continue
        
        # 初始化輸入
        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": ""
        }
        
        # 執行 Graph
        result = app.invoke(inputs)
        
        # 輸出最終結果
        print("\n========= 最終結果 =========")
        print(f"原文: {result['original_text']}")
        print(f"最終翻譯: {result['translated_text']}")
        print(f"最終次數: {result['attempts']}")