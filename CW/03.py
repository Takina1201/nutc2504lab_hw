"""
課堂作業-03：RAG + Query ReWrite
==================================
1. 把 data_01~05.txt 切塊後嵌入到 VDB
2. 實作 Query_ReWrite（多輪對話 → 獨立搜尋語句）
3. 用 Query_ReWrite 的結果去 Retrieval
4. 結合 LLM 去回答
5. 完成 Re_Write_questions.csv 和 questions.csv
"""

import os
import re
import csv
import time
import json
import requests
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# 設定
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR  # data 檔案在同一層

# API
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "google/gemma-3-27b-it"

# 切塊參數
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# Qdrant
COLLECTION_NAME = "cw03_chunks"


# ============================================================
# 工具函數
# ============================================================
def get_embedding(texts: list[str]) -> tuple:
    """使用 Embedding API 取得文本向量"""
    data = {
        "texts": texts,
        "task_description": "檢索技術文件",
        "normalize": True,
    }
    try:
        resp = requests.post(EMBED_API_URL, json=data, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            return result["embeddings"], result["dimension"]
        print(f"  ❌ Embedding API 錯誤: {resp.status_code} - {resp.text}")
        return None, None
    except Exception as e:
        print(f"  ❌ Embedding 連線失敗: {e}")
        return None, None


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """呼叫 LLM API"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(LLM_API_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"].strip()
        print(f"  ❌ LLM API 錯誤: {resp.status_code} - {resp.text}")
        return ""
    except Exception as e:
        print(f"  ❌ LLM 連線失敗: {e}")
        return ""


def call_llm_multi_turn(messages: list[dict], temperature: float = 0.3) -> str:
    """呼叫 LLM API（多輪對話）"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    try:
        resp = requests.post(LLM_API_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  ❌ LLM API 錯誤: {resp.status_code}")
        return ""
    except Exception as e:
        print(f"  ❌ LLM 連線失敗: {e}")
        return ""


def read_csv(path: str) -> list[dict]:
    """讀取 CSV"""
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict], fieldnames: list[str]):
    """寫入 CSV (utf-8-sig)"""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ CSV 已儲存：{path}（{len(rows)} 筆）")


# ============================================================
# Step 1：讀取資料 → 切塊 → 嵌入 VDB
# ============================================================
def load_and_chunk_data() -> list[dict]:
    """讀取 data_01~05.txt 並進行滑動視窗切塊"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks = []
    for i in range(1, 6):
        filename = f"data_0{i}.txt"
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ⚠️ {filename} 不存在，跳過")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_chunks.append({"text": chunk, "source": filename})
        print(f"  ✅ {filename}：{len(content)} 字元 → {len(chunks)} 塊")

    return all_chunks


def build_vdb(client: QdrantClient, chunks: list[dict], dim: int):
    """建立 Qdrant Collection 並嵌入所有切塊"""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    all_points = []
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embs, _ = get_embedding([c["text"] for c in batch])
        if embs is None:
            continue
        for j, (chunk, emb) in enumerate(zip(batch, embs)):
            all_points.append(PointStruct(
                id=i + j, vector=emb,
                payload={"text": chunk["text"], "source": chunk["source"]},
            ))
        time.sleep(0.3)

    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"  ✅ {COLLECTION_NAME}：{len(all_points)} 個向量已嵌入")


def retrieve(client: QdrantClient, query: str, top_k: int = 3) -> list[dict]:
    """從 VDB 檢索最相關的切塊"""
    emb, _ = get_embedding([query])
    if emb is None:
        return []

    results = client.query_points(
        collection_name=COLLECTION_NAME, query=emb[0], limit=top_k
    )

    return [
        {"text": p.payload["text"], "source": p.payload["source"], "score": p.score}
        for p in results.points
    ]


# ============================================================
# Step 2：Query ReWrite
# ============================================================
def load_rewrite_prompt() -> str:
    """讀取 Prompt_ReWrite.txt"""
    path = os.path.join(DATA_DIR, "Prompt_ReWrite.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    # 備用 prompt
    return """你是一個 RAG 系統的查詢重寫專家。
將使用者的「最新問題」結合「對話歷史」，重寫成適合向量資料庫搜尋的「獨立搜尋語句」。
規則：1.指代消解 2.補全上下文 3.保留原意 4.關鍵字增強 5.繁體中文
直接輸出重寫後的語句，不要任何解釋。"""


def query_rewrite(rewrite_prompt: str, conversation_history: list[dict],
                  current_question: str) -> str:
    """
    使用 LLM 將多輪對話中的問題重寫為獨立搜尋語句
    conversation_history: [{"role": "user"/"assistant", "content": "..."}, ...]
    """
    # 組合對話歷史成文字
    history_text = ""
    if conversation_history:
        history_text = "【對話歷史】\n"
        for msg in conversation_history:
            role = "使用者" if msg["role"] == "user" else "助理"
            history_text += f"{role}：{msg['content']}\n"
        history_text += "\n"

    user_msg = f"""{history_text}【最新問題】
{current_question}

請將上述最新問題重寫為一個獨立的搜尋語句："""

    rewritten = call_llm(rewrite_prompt, user_msg, temperature=0.1)
    # 清理可能的多餘文字
    rewritten = rewritten.strip().strip('"').strip("'")
    return rewritten


# ============================================================
# Step 3 & 4：RAG 回答（檢索 + LLM 生成）
# ============================================================
RAG_SYSTEM_PROMPT = """你是一個專業的問答助理。請根據以下「參考資料」回答使用者的問題。
規則：
1. 只根據參考資料中的內容回答，不要自行編造
2. 回答要簡潔精確，直接回答問題重點
3. 使用繁體中文回答"""


def rag_answer(client: QdrantClient, question: str, top_k: int = 3) -> tuple:
    """
    RAG 流程：檢索 → 組合 context → LLM 回答
    回傳: (answer, source, retrieved_texts)
    """
    results = retrieve(client, question, top_k)
    if not results:
        return "無法檢索到相關資料", "", []

    # 組合 context
    context = ""
    sources = set()
    for i, r in enumerate(results):
        context += f"【段落 {i+1}】（來源：{r['source']}，相似度：{r['score']:.4f}）\n{r['text']}\n\n"
        sources.add(r["source"])

    # 主要來源（分數最高的）
    main_source = results[0]["source"]

    user_msg = f"""【參考資料】
{context}

【問題】
{question}

請根據參考資料回答上述問題："""

    answer = call_llm(RAG_SYSTEM_PROMPT, user_msg, temperature=0.2)
    return answer, main_source, results


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 60)
    print("課堂作業-03：RAG + Query ReWrite")
    print("=" * 60)

    # ─── Step 1：切塊 & 嵌入 VDB ───
    print("\n📦 Step 1：讀取資料 → 切塊 → 嵌入 VDB")
    print("-" * 40)

    chunks = load_and_chunk_data()
    if not chunks:
        print("❌ 找不到資料檔案！")
        return

    print(f"\n  總計 {len(chunks)} 個切塊")

    # 取得維度
    _, dim = get_embedding(["測試"])
    if dim is None:
        print("❌ Embedding API 不可用")
        return
    print(f"  向量維度：{dim}")

    # 建立 VDB
    client = QdrantClient(url="http://localhost:6333")
    build_vdb(client, chunks, dim)

    # ─── Step 2 & 3：處理 questions.csv（直接檢索） ───
    print("\n📋 Step 2：處理 questions.csv（直接 RAG）")
    print("-" * 40)

    q_path = os.path.join(DATA_DIR, "questions.csv")
    questions = read_csv(q_path)

    q_results = []
    for q in questions:
        q_id = q["題目_ID"]
        q_text = q["題目"]
        print(f"\n  Q{q_id}: {q_text[:45]}...")

        answer, source, hits = rag_answer(client, q_text)
        print(f"      答案: {answer[:60]}...")
        print(f"      來源: {source}")

        q_results.append({
            "題目_ID": q_id,
            "題目": q_text,
            "標準答案": answer,
            "來源文件": source,
        })
        time.sleep(0.5)

    # 輸出 questions_answer.csv
    q_out_path = os.path.join(DATA_DIR, "questions_answer.csv")
    write_csv(q_out_path, q_results, ["題目_ID", "題目", "標準答案", "來源文件"])

    # ─── Step 4 & 5：處理 Re_Write_questions.csv（Query ReWrite + RAG） ───
    print("\n🔄 Step 3：處理 Re_Write_questions.csv（Query ReWrite + RAG）")
    print("-" * 40)

    rewrite_prompt = load_rewrite_prompt()
    print("  ✅ 已載入 Prompt_ReWrite.txt")

    rw_path = os.path.join(DATA_DIR, "Re_Write_questions.csv")
    rw_questions = read_csv(rw_path)

    # 按 conversation_id 分組
    conversations = {}
    for q in rw_questions:
        conv_id = q["conversation_id"]
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(q)

    rw_results = []

    for conv_id, conv_questions in conversations.items():
        print(f"\n  === 對話 {conv_id} ===")
        conversation_history = []  # 累積對話歷史

        for q in conv_questions:
            q_id = q["questions_id"]
            q_text = q["questions"]
            print(f"\n    Q{conv_id}-{q_id}: {q_text}")

            # Step A：Query ReWrite
            if len(conversation_history) > 0:
                # 有對話歷史 → 需要重寫
                rewritten = query_rewrite(rewrite_prompt, conversation_history, q_text)
                print(f"    ✏️  重寫後: {rewritten}")
            else:
                # 第一個問題 → 不需要重寫
                rewritten = q_text
                print(f"    ✏️  首題，無需重寫")

            # Step B：用重寫後的 query 檢索
            answer, source, hits = rag_answer(client, rewritten)
            print(f"    💬 答案: {answer[:60]}...")
            print(f"    📂 來源: {source}")

            rw_results.append({
                "conversation_id": conv_id,
                "questions_id": q_id,
                "questions": q_text,
                "answer": answer,
                "source": source,
            })

            # 累積對話歷史
            conversation_history.append({"role": "user", "content": q_text})
            conversation_history.append({"role": "assistant", "content": answer})

            time.sleep(0.5)

    # 輸出 Re_Write_answer.csv
    rw_out_path = os.path.join(DATA_DIR, "Re_Write_answer.csv")
    write_csv(rw_out_path, rw_results,
              ["conversation_id", "questions_id", "questions", "answer", "source"])

    # ─── 完成 ───
    print(f"""
{'=' * 60}
✅ 課堂作業-03 完成！
{'=' * 60}

📊 執行摘要：
  1. ✅ 資料切塊：{len(chunks)} 塊 → Qdrant VDB
  2. ✅ questions.csv：{len(q_results)} 題已回答 → questions_answer.csv
  3. ✅ Re_Write_questions.csv：{len(rw_results)} 題已重寫+回答 → Re_Write_answer.csv

📁 輸出檔案：
  - {q_out_path}
  - {rw_out_path}

📌 上傳到 GitHub (CW/03/)
""")


if __name__ == "__main__":
    main()