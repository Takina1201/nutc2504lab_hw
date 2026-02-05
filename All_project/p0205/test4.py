"""
LangGraph 效能優化與架構模式 - 實作(ch7-1)
Cache 機制的翻譯機 (基於 ch6-2 的主題)

特點：
1. 加入快取機制，相同的原文不需要重複翻譯
2. 快取命中時直接回傳結果，節省 API 呼叫
3. 快取未命中時才進行翻譯流程
4. 翻譯完成後自動存入快取

流程圖：
    __start__
        |
        v
    check_cache
        |
   [cache_router]
    /         \\
   v           v
 END      translator  <---+
(命中)        |           |
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
import random
import time
import json
import os
from typing import Annotated, TypedDict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# 配置與快取函式
# ============================================================

llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

CACHE_FILE = "translation_cache.json"


def load_cache():
    """載入快取檔案"""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_cache(original: str, translated: str):
    """儲存翻譯結果到快取"""
    data = load_cache()
    data[original] = translated
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================
# 1. 定義狀態
# ============================================================

class State(TypedDict):
    original_text: str       # 原始文字
    translated_text: str     # 翻譯結果
    critique: str            # 評語
    attempts: int            # 重試次數
    is_cache_hit: bool       # 標記是否命中快取


# ============================================================
# 2. 定義節點
# ============================================================

def check_cache_node(state: State):
    """檢查快取節點"""
    print("\n--- 檢查快取 (Check Cache) ---")
    data = load_cache()
    original = state["original_text"]
    
    if original in data:
        print("命中快取！直接回傳結果。")
        return {
            "translated_text": data[original],
            "is_cache_hit": True
        }
    else:
        print("未命中快取，準備開始翻譯流程...")
        return {"is_cache_hit": False}


def translator_node(state: State):
    """翻譯節點"""
    print(f"\n--- 翻譯嘗試（第 {state['attempts'] + 1} 次）---")
    
    prompt = f"你是一名翻譯員，請將以下中文翻譯成英文，不須任何解釋: '{state['original_text']}'"
    
    if state['critique']:
        prompt += f"\n\n上一輪的審查意見是: {state['critique']}。請根據意見修正翻譯。"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "translated_text": response.content,
        "attempts": state['attempts'] + 1
    }


def reflector_node(state: State):
    """審查節點"""
    print("--- 審查中 (Reflection) ---")
    
    prompt = f"原文: {state['original_text']}\n翻譯: {state['translated_text']}\n請檢查翻譯是否準確且通順。如果翻譯很完美，請只回覆 'PASS'。如果需要修改，請給出簡短的具體建議。"
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"critique": response.content}


# ============================================================
# 3. 定義路由 (Routers)
# ============================================================

def cache_router(state: State) -> Literal["end", "translator"]:
    """[新增] 快取路由：有快取就結束，沒快取就去翻譯"""
    if state["is_cache_hit"]:
        return "end"
    return "translator"


def critique_router(state: State) -> Literal["translator", "end"]:
    """審查路由"""
    if "PASS" in state['critique'].upper():
        print("--- 審查通過！---")
        return "end"
    elif state['attempts'] >= 3:
        print("--- 達到最大重試次數 ---")
        return "end"
    else:
        print(f"--- 退回重寫: {state['critique']} ---")
        return "translator"


# ============================================================
# 4. 組裝 Graph
# ============================================================

workflow = StateGraph(State)

# 加入節點
workflow.add_node("check_cache", check_cache_node)  # 新節點
workflow.add_node("translator", translator_node)
workflow.add_node("reflector", reflector_node)

# 一律先走 check_cache
workflow.set_entry_point("check_cache")

# 設定快取後的路徑 (Cache Hit -> END; Cache Miss -> Translator)
workflow.add_conditional_edges(
    "check_cache",
    cache_router,
    {
        "end": END,
        "translator": "translator"
    }
)

# 正常的翻譯迴圈路徑
workflow.add_edge("translator", "reflector")
workflow.add_conditional_edges(
    "reflector",
    critique_router,
    {"translator": "translator", "end": END}
)

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print(f"快取檔案: {CACHE_FILE}")
    
    print("\n" + "=" * 60)
    print("LangGraph 效能優化 - Cache 機制翻譯機 (ch7-1)")
    print("=" * 60)
    print("相同的原文會直接從快取回傳，不需重複翻譯")
    print("輸入 'exit' 或 'q' 離開")
    print("=" * 60)
    
    while True:
        user_input = input("\n請輸入要翻譯的中文 (exit/q 離開): ")
        if user_input.lower() in ["exit", "q"]:
            break
        
        if not user_input.strip():
            continue
        
        inputs = {
            "original_text": user_input,
            "attempts": 0,
            "critique": "",
            "is_cache_hit": False,
            "translated_text": ""  # 初始為空
        }
        
        # 執行 Graph
        result = app.invoke(inputs)
        
        # 如果不是從快取來的（代表是新算出來的），就寫入快取
        if not result["is_cache_hit"]:
            save_cache(result["original_text"], result["translated_text"])
            print("(已將新翻譯寫入快取)")
        
        # 輸出結果
        print("\n========= 最終結果 =========")
        print(f"原文: {result['original_text']}")
        print(f"翻譯: {result['translated_text']}")
        print(f"來源: {'快取 (Cache)' if result['is_cache_hit'] else '生成 (LLM)'}")