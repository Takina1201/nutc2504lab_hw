import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# ============================================================
# 1. 設定與 API 初始化
# ============================================================

API_KEY = ""
BASE_URL = "https://ws-03.wade0426.me/v1"
LLM_MODEL = "/models/gpt-oss-120b"

# 初始化 LLM
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=LLM_MODEL,
    temperature=0.1
)

# ============================================================
# 2. 定義 State (狀態)
# ============================================================

class MeetingState(TypedDict):
    srt_file_path: str   # SRT 檔案路徑
    raw_content: str     # 讀取進來的 SRT 原始文字
    transcript: str      # 整理後的逐字稿 (Minutes Taker)
    summary: str         # 重點摘要 (Summarizer)
    final_report: str    # 最終整合報告 (Writer)

# ============================================================
# 3. 定義節點 (Nodes)
# ============================================================

def asr_node(state: MeetingState) -> dict:
    """
    asr 節點：負責讀取已經跑好的 SRT 檔案
    """
    file_path = state['srt_file_path']
    print(f"\n[Loader] 正在讀取 SRT 檔案: {file_path} ...")
    
    if not os.path.exists(file_path):
        return {"raw_content": "錯誤：找不到 SRT 檔案"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[Loader] 讀取成功！內容長度：{len(content)} 字元")
        return {"raw_content": content}
    except Exception as e:
        return {"raw_content": f"讀取失敗: {e}"}

def minutes_taker_node(state: MeetingState) -> dict:
    """
    逐字稿節點：將 SRT 格式轉換為漂亮的對話紀錄
    """
    print("[Minutes Taker] 正在整理詳細逐字稿 (SRT 格式轉換)...")
    raw_content = state["raw_content"]
    
    # 使用 Prompt 讓 LLM 幫我們整理 SRT 格式
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位專業的會議記錄員。
        你將收到一份 SRT 字幕檔內容，請幫我整理成「詳細逐字稿」。
        
        處理規則：
        1. 保留時間戳記，格式簡化為 [HH:MM:SS-HH:MM:SS]。
        2. 辨識講者：根據內容，講者是「主持人 (好序列好哥)」。
        3. 將斷裂的語句合併，使其通順。
        4. 輸出格式範例：
           **時間** | **發言內容**
           ------- | -------
           00:00:00-00:00:03 | 主持人：歡迎來到天下文化 podcast...
           00:00:03-00:00:10 |今天要介紹這本書...
        """),
        ("user", "{text}")
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": raw_content})
        return {"transcript": result}
    except Exception as e:
        return {"transcript": f"處理失敗: {e}"}

def summarizer_node(state: MeetingState) -> dict:
    """
    摘要節點：提取重點摘要與行動項目
    """
    print("[Summarizer] 正在撰寫重點摘要...")
    raw_content = state["raw_content"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位會議記錄助手。請閱讀以下 SRT 字幕內容，整理出「重點摘要報告」。
        
        格式要求：
        1. **書籍資訊**: 書名、作者。
        2. **決策結果**: 用一句話總結核心主旨。
        3. **待辦事項**: 條列式列出講者提到的 3 個重點。
        4. 書本案例補充
        5. 使用繁體中文。
        """),
        ("user", "{text}")
    ])
    
    try:
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"text": raw_content})
        return {"summary": result}
    except Exception as e:
        return {"summary": f"處理失敗: {e}"}

def writer_node(state: MeetingState) -> dict:
    """
    整合節點：將兩份資料合併輸出
    """
    print("[Writer] 正在整合最終報告...")
    
    final_output = f"""
##################################################
#              智慧會議記錄報告                  #
##################################################

【一、重點摘要】
{state['summary']}

--------------------------------------------------

【二、詳細記錄(Detailed Minutes)】
{state['transcript']}

##################################################
    """
    return {"final_report": final_output}

# ============================================================
# 4. 建立 Graph (流程圖)
# ============================================================

workflow = StateGraph(MeetingState)

# 新增節點
workflow.add_node("asr", asr_node)
workflow.add_node("minutes_taker", minutes_taker_node)
workflow.add_node("summarizer", summarizer_node)
workflow.add_node("writer", writer_node)

# 設定進入點
workflow.set_entry_point("asr")

# 設定邊 (並行處理)
# 讀取完檔案後，同時送給「逐字稿整理」和「摘要總結」
workflow.add_edge("asr", "minutes_taker")
workflow.add_edge("asr", "summarizer")

# 兩者都完成後，匯聚到 Writer
workflow.add_edge("minutes_taker", "writer")
workflow.add_edge("summarizer", "writer")

# 結束
workflow.add_edge("writer", END)

# 編譯 Graph
app = workflow.compile()
print(app.get_graph().draw_ascii())
# ============================================================
# 5. 執行
# ============================================================

if __name__ == "__main__":
    # 修正：直接指向檔案的絕對路徑 (根據你的截圖路徑)
    # 使用 r"..." 代表原始字串，避免 Windows 路徑的反斜線問題
    target_file = r"d:\nutc2504lab_hw\All_project\p0204\out\97.srt"
    # 檢查檔案是否存在
    if os.path.exists(target_file):
        input_data = {"srt_file_path": target_file}
        
        # 執行 LangGraph
        try:
            result = app.invoke(input_data)
            print("\n" + "="*20 + " 執行結果 " + "="*20)
            print(result["final_report"])
        except Exception as e:
            print(f"❌ 執行過程中發生錯誤: {e}")
            
    else:
        print(f"\n❌ 錯誤：仍然找不到檔案")
        print(f"請確認該路徑下是否真的有 97.srt 檔案：\n{target_file}")