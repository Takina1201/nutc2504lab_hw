"""
LangGraph 效能優化與架構模式 - 實作(ch7-2)
混合架構效能優化的 QA Chat

特點：
1. 快取機制：相同問題直接回傳快取結果
2. 快速通道 (Fast Track)：簡單打招呼用小模型快速回覆
3. 專家模式 (Expert)：複雜問題用大模型並使用流式傳輸

三種來源：
- CACHE: 從快取讀取（最快）
- FAST_TRACK_API: 用小模型快速回覆（快）
- LLM_EXPERT: 用大模型詳細回答（慢但詳細）

流程圖：
    __start__
        |
        v
    check_cache
        |
   [master_router]
    /    |    \\
   v     v     v
 END  fast_bot  expert_bot
(快取)   |         |
         v         v
       __end__  __end__

請先安裝: pip install langgraph langchain langchain-openai

API 配置：
- 大模型: https://ws-02.wade0426.me/v1 (google/gemma-3-27b-it)
- 小模型: https://ws-05.huannago.com/v1 (Qwen3-VL-8B-Instruct-BF16.gguf)
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

# 主要 LLM（大模型，用於專家回答）
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="",
    model="google/gemma-3-27b-it",
    temperature=0.7
)

# [新增] 快速通道專用的模型實例
# 指向你指定的 ws-05 URL
fast_llm = ChatOpenAI(
    model="Qwen3-VL-8B-Instruct-BF16.gguf",
    api_key="",  # 假設 Key 通用
    base_url="https://ws-05.huannago.com/v1",
    temperature=0
)

# 設定快取檔案名稱
CACHE_FILE = "qa_cache.json"


# ============================================================
# 工具函式（維持原樣）
# ============================================================

def get_clean_key(text: str) -> str:
    """統一將問題標準化"""
    return text.replace(" ", "").replace("?", "").replace("？", "")


def load_cache():
    """從 JSON 讀取快取資料"""
    if not os.path.exists(CACHE_FILE):
        # 初始化預設快取
        default_data = {
            get_clean_key("LangGraph是什麼"): "LangGraph 是一個用於構建有狀態、多參與者應用程式的框架",
            get_clean_key("你的名字"): "我是這個課程的 AI 助教。"
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)
        return default_data
    
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_cache(new_data: dict):
    """將資料寫入 JSON"""
    current_data = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                current_data = json.load(f)
        except:
            pass
    current_data.update(new_data)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(current_data, f, ensure_ascii=False, indent=4)


# ============================================================
# LangGraph 定義區
# ============================================================

# 1. 定義狀態
class State(TypedDict):
    question: str
    answer: str
    source: str  # CACHE / FAST_TRACK_API / LLM_EXPERT


# ============================================================
# 2. 定義節點
# ============================================================

def check_cache_node(state: State):
    """檢查快取"""
    print(f"\n[系統] 收到問題: {state['question']}")
    cache_data = load_cache()
    clean_query = get_clean_key(state['question'])
    
    if clean_query in cache_data:
        print("--- 命中快取 (Cache Hit) ---")
        return {
            "answer": cache_data[clean_query],
            "source": "CACHE"
        }
    else:
        print("--- 快取未命中 (Cache Miss) ---")
        return {"source": "MISS"}


def fast_reply_node(state: State):
    """快速通道：使用小模型快速回覆"""
    print("--- 進入快速通道 (Fast Track API) ---")
    
    response = fast_llm.invoke([HumanMessage(content=state['question'])])
    
    return {
        "answer": response.content,
        "source": "FAST_TRACK_API"
    }


def expert_node(state: State):
    """
    慢速通道：呼叫 LLM 並使用「流式傳輸」
    """
    print("--- 進入專家模式 (LLM Expert) ---")
    
    prompt = f"請以專業的角度回答以下問題: {state['question']}"
    
    # 使用流式傳輸
    chunks = llm.stream([HumanMessage(content=prompt)])
    
    full_answer = ""
    print("AI 正在思考並打字: ", end="", flush=True)
    
    for chunk in chunks:
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_answer += content
    print("\n")
    
    # 將完整回答存入快取
    clean_key = get_clean_key(state['question'])
    save_cache({clean_key: full_answer})
    print(f"--- [系統] 已將完整回答寫入 {CACHE_FILE} ---")
    
    return {
        "answer": full_answer,
        "source": "LLM_EXPERT"
    }


# ============================================================
# 3. 定義路由
# ============================================================

def master_router(state: State):
    """主路由控制器"""
    # 如果已經有答案（快取命中），直接結束
    if state.get("answer"):
        return "end"
    
    question = state['question']
    
    # 只要有這些打招呼的詞，就走 Fast Track (ws-05)
    if any(word in question for word in ["你好", "嗨", "早安", "哈囉"]):
        return "fast"
    else:
        return "expert"


# ============================================================
# 4. 組裝 Graph
# ============================================================

workflow = StateGraph(State)

# 加入節點
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("fast_bot", fast_reply_node)
workflow.add_node("expert_bot", expert_node)

# 設定入口
workflow.set_entry_point("check_cache")

# 設定條件邊
workflow.add_conditional_edges(
    "check_cache",
    master_router,
    {
        "end": END,
        "fast": "fast_bot",
        "expert": "expert_bot"
    }
)

# fast_bot 和 expert_bot 完成後都結束
workflow.add_edge("fast_bot", END)
workflow.add_edge("expert_bot", END)

# 編譯
app = workflow.compile()
print(app.get_graph().draw_ascii())


# ============================================================
# 測試執行
# ============================================================

if __name__ == "__main__":
    print(f"快取檔案將儲存於: {os.path.abspath(CACHE_FILE)}")
    print("提示：試著輸入 '你好' 測試 Fast API；輸入專業問題測試 Expert API。")
    
    print("\n" + "=" * 60)
    print("LangGraph 效能優化 - 混合架構 QA Chat (ch7-2)")
    print("=" * 60)
    
    while True:
        user_input = input("\n請輸入問題（輸入 q 離開）: ")
        if user_input.lower() == 'q':
            break
        
        if not user_input.strip():
            continue
        
        inputs = {"question": user_input}
        
        start_time = time.time()
        try:
            result = app.invoke(inputs)
            end_time = time.time()
            
            print("-" * 30)
            print(f"來源: [{result['source']}]")
            print(f"耗時: {end_time - start_time:.4f} 秒")
            
            # 只有非 Expert (Cache 或 Fast API) 需要在這裡印出結果
            if result['source'] != "LLM_EXPERT":
                print(f"回答: {result['answer']}")
            else:
                print("(回答已於上方流式輸出完畢)")
                
        except Exception as e:
            print(f"發生錯誤: {e}")