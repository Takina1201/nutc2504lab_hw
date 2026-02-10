"""
課堂作業-04：RAG + Hybrid Search + ReRank + Query ReWrite
==========================================================
流程：
  1. data_01~05.txt → 滑動視窗切塊 → Embedding → Qdrant VDB
  2. 建立 BM25 索引（關鍵字搜尋）
  3. Hybrid Search = Dense(向量) + Sparse(BM25) → RRF 融合
  4. ReRank：LLM 判斷相關性重新排序
  5. LLM 從 Top-3 萃取精準答案
  6. Re_Write：多輪對話 Query Rewrite → 同上流程
  7. 輸出 questions_answer.csv、Re_Write_answer.csv
"""

import os
import re
import csv
import json
import time
import math
import requests
import numpy as np
import jieba
from collections import Counter
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# 設定
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# API
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "google/gemma-3-27b-it"

# 切塊參數
CHUNK_SIZE = 300
CHUNK_OVERLAP = 100

# 檢索參數
DENSE_TOP_K = 10       # 向量搜尋取前 10
BM25_TOP_K = 10        # BM25 搜尋取前 10
HYBRID_TOP_K = 10      # RRF 融合後取前 10
RERANK_TOP_K = 3       # ReRank 後取前 3
RRF_K = 60             # RRF 常數

# Collection 名稱
COLLECTION_NAME = "cw04_chunks"


# ============================================================
# 工具函數
# ============================================================
def get_embedding(texts: list[str]) -> tuple:
    """Embedding API"""
    data = {"texts": texts, "normalize": True, "batch_size": 32}
    try:
        resp = requests.post(EMBED_API_URL, json=data, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            return result["embeddings"], result["dimension"]
        print(f"  ❌ Embedding API 錯誤: {resp.status_code}")
        return None, None
    except Exception as e:
        print(f"  ❌ Embedding API 連線失敗: {e}")
        return None, None


def call_llm(system_prompt: str, user_prompt: str,
             temperature: float = 0.1, max_tokens: int = 512) -> str:
    """LLM API"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(LLM_API_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  ❌ LLM API 錯誤: {resp.status_code} - {resp.text[:200]}")
        return ""
    except Exception as e:
        print(f"  ❌ LLM 連線失敗: {e}")
        return ""


def call_llm_messages(messages: list[dict],
                      temperature: float = 0.1, max_tokens: int = 256) -> str:
    """LLM API（自訂 messages）"""
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
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


# ============================================================
# 文本切塊
# ============================================================
def chunk_texts(data_files: dict) -> list[dict]:
    """滑動視窗切塊"""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = []
    for filename, content in data_files.items():
        texts = splitter.split_text(content)
        for t in texts:
            chunks.append({"text": t, "source": filename})
        print(f"  ✅ {filename}：{len(texts)} 塊")
    return chunks


# ============================================================
# BM25 索引
# ============================================================
def tokenize_chinese(text: str) -> list[str]:
    """中文分詞（jieba）"""
    words = jieba.lcut(text)
    # 過濾停用詞和單字元
    stop_words = {"的", "了", "在", "是", "我", "有", "和", "就",
                  "不", "人", "都", "一", "一個", "上", "也", "很",
                  "到", "說", "要", "去", "你", "會", "著", "沒有",
                  "看", "好", "自己", "這", "他", "她", "它", "們",
                  "那", "被", "從", "對", "為", "與", "等", "但",
                  "而", "及", "或", "之", "其", "中", "所", "以",
                  "可", "能", "將", "還", "因", "此", "則", "如",
                  "於", "個", "每", "又", "把", "讓", "用", "做"}
    return [w for w in words if len(w) > 1 and w not in stop_words]


class BM25Index:
    """BM25 關鍵字搜尋索引"""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized = [tokenize_chinese(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)
        print(f"  ✅ BM25 索引建立完成：{len(chunks)} 個文檔")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[dict]:
        """BM25 搜尋"""
        query_tokens = tokenize_chinese(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "index": int(idx),
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "bm25_score": float(scores[idx]),
                })
        return results


# ============================================================
# Hybrid Search（RRF 融合）
# ============================================================
def reciprocal_rank_fusion(dense_results: list[dict],
                           bm25_results: list[dict],
                           k: int = RRF_K,
                           top_k: int = HYBRID_TOP_K) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF)
    score(d) = Σ 1/(k + rank_i(d))
    """
    # 建立文檔 → RRF 分數的映射（用 chunk index 作為 key）
    rrf_scores = {}
    doc_info = {}

    # Dense results
    for rank, hit in enumerate(dense_results):
        idx = hit["index"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        doc_info[idx] = {"text": hit["text"], "source": hit["source"]}

    # BM25 results
    for rank, hit in enumerate(bm25_results):
        idx = hit["index"]
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        doc_info[idx] = {"text": hit["text"], "source": hit["source"]}

    # 排序
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in sorted_docs[:top_k]:
        results.append({
            "index": idx,
            "text": doc_info[idx]["text"],
            "source": doc_info[idx]["source"],
            "rrf_score": score,
        })
    return results


# ============================================================
# ReRank（使用 LLM 判斷相關性）
# ============================================================
RERANK_SYSTEM_PROMPT = """你是一個文件相關性評估專家。請判斷給定的「文件段落」與「查詢問題」的相關程度。

評分標準（0-10 分）：
- 9-10：文件直接且完整地回答了問題
- 7-8：文件包含回答問題所需的大部分關鍵資訊
- 5-6：文件部分相關，包含一些有用資訊
- 3-4：文件略有相關，但主要內容不同
- 0-2：文件與問題完全無關

請只輸出一個數字（0-10），不要有任何其他文字。"""


def rerank_with_llm(query: str, candidates: list[dict],
                    top_k: int = RERANK_TOP_K) -> list[dict]:
    """使用 LLM 對候選文檔進行 ReRank"""
    scored = []

    for cand in candidates:
        user_prompt = f"【查詢問題】\n{query}\n\n【文件段落】\n{cand['text']}"
        response = call_llm(RERANK_SYSTEM_PROMPT, user_prompt,
                            temperature=0.0, max_tokens=16)

        # 解析分數
        try:
            # 嘗試從回應中提取數字
            nums = re.findall(r'\d+(?:\.\d+)?', response)
            score = float(nums[0]) if nums else 5.0
            score = min(max(score, 0), 10)
        except (ValueError, IndexError):
            score = 5.0

        scored.append({
            **cand,
            "rerank_score": score,
        })
        time.sleep(0.1)

    # 按 ReRank 分數排序
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


# ============================================================
# LLM 答案生成
# ============================================================
ANSWER_SYSTEM_PROMPT = """你是一個精準的問答助理。請根據提供的「參考段落」回答問題。

規則：
1. 只根據參考段落中的內容回答，不要編造
2. 回答要簡潔、精確、完整，直接回答問題的重點
3. 包含所有相關的關鍵數據、名稱、細節
4. 使用繁體中文
5. 不要加上「根據參考段落」等前綴，直接回答"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """從 Top-K chunks 用 LLM 萃取精準答案"""
    context = ""
    for i, c in enumerate(chunks):
        context += f"【段落 {i+1}】（{c['source']}）\n{c['text']}\n\n"

    user_prompt = f"【參考段落】\n{context}\n【問題】\n{question}\n\n請根據參考段落精準回答上述問題："

    return call_llm(ANSWER_SYSTEM_PROMPT, user_prompt)


# ============================================================
# Query ReWrite（多輪對話查詢重寫）
# ============================================================
def rewrite_query(prompt_rewrite: str, conversation_history: list[dict],
                  current_question: str) -> str:
    """使用 Prompt_ReWrite.txt 重寫多輪對話中的查詢"""
    # 組建歷史對話文字
    history_text = ""
    for msg in conversation_history:
        role = "使用者" if msg["role"] == "user" else "助理"
        history_text += f"{role}：{msg['content']}\n"

    user_prompt = f"【對話歷史】\n{history_text}\n【最新問題】\n{current_question}"

    rewritten = call_llm(prompt_rewrite, user_prompt,
                         temperature=0.1, max_tokens=256)
    return rewritten.strip() if rewritten else current_question


# ============================================================
# 完整 RAG Pipeline
# ============================================================
def rag_pipeline(query: str, qdrant_client: QdrantClient,
                 bm25_index: BM25Index, chunks: list[dict],
                 label: str = "") -> tuple[str, str]:
    """
    完整 RAG 流程：Hybrid Search → ReRank → LLM Answer
    回傳 (answer, source)
    """
    prefix = f"    [{label}] " if label else "    "

    # Step 1: Dense Search（向量搜尋）
    emb, _ = get_embedding([query])
    if emb is None:
        return "", ""

    dense_raw = qdrant_client.query_points(
        collection_name=COLLECTION_NAME, query=emb[0], limit=DENSE_TOP_K
    )
    dense_results = []
    for p in dense_raw.points:
        # 找到對應的 chunk index
        idx = p.id
        dense_results.append({
            "index": idx,
            "text": p.payload["text"],
            "source": p.payload["source"],
            "dense_score": p.score,
        })

    # Step 2: BM25 Search（關鍵字搜尋）
    bm25_results = bm25_index.search(query, BM25_TOP_K)

    print(f"{prefix}Dense: {len(dense_results)} 筆 | BM25: {len(bm25_results)} 筆", end="")

    # Step 3: Hybrid Fusion（RRF）
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results,
                                            top_k=HYBRID_TOP_K)
    print(f" → RRF: {len(hybrid_results)} 筆", end="")

    # Step 4: ReRank（LLM 重新排序）
    reranked = rerank_with_llm(query, hybrid_results, RERANK_TOP_K)
    print(f" → ReRank Top-{len(reranked)}", end="")

    # Step 5: LLM Answer Generation
    answer = generate_answer(query, reranked)
    source = reranked[0]["source"] if reranked else ""
    print(f" → ✅")

    return answer, source


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 65)
    print("課堂作業-04：RAG + Hybrid Search + ReRank + Query ReWrite")
    print("=" * 65)

    # ── 1. 讀取資料 ──
    print("\n📂 步驟 1：讀取資料檔案")
    print("-" * 40)

    data_files = {}
    for i in range(1, 6):
        fn = f"data_{i:02d}.txt"
        path = os.path.join(SCRIPT_DIR, fn)
        if not os.path.exists(path):
            # 嘗試其他路徑
            for alt in [os.path.join(SCRIPT_DIR, "data", fn),
                        os.path.join(SCRIPT_DIR, "..", fn)]:
                if os.path.exists(alt):
                    path = alt
                    break
        with open(path, "r", encoding="utf-8") as f:
            data_files[fn] = f.read()
        print(f"  ✅ {fn}：{len(data_files[fn])} 字元")

    # 讀取 Prompt_ReWrite.txt
    prompt_rewrite_path = os.path.join(SCRIPT_DIR, "Prompt_ReWrite.txt")
    if not os.path.exists(prompt_rewrite_path):
        for alt in [os.path.join(SCRIPT_DIR, "..", "Prompt_ReWrite.txt")]:
            if os.path.exists(alt):
                prompt_rewrite_path = alt
                break
    with open(prompt_rewrite_path, "r", encoding="utf-8") as f:
        prompt_rewrite = f.read()
    print(f"  ✅ Prompt_ReWrite.txt：{len(prompt_rewrite)} 字元")

    # 讀取 questions.csv
    q_path = os.path.join(SCRIPT_DIR, "questions.csv")
    if not os.path.exists(q_path):
        for alt in [os.path.join(SCRIPT_DIR, "..", "questions.csv")]:
            if os.path.exists(alt):
                q_path = alt
                break
    with open(q_path, "r", encoding="utf-8-sig") as f:
        questions = list(csv.DictReader(f))
    print(f"  ✅ questions.csv：{len(questions)} 題")

    # 讀取 Re_Write_questions.csv
    rw_path = os.path.join(SCRIPT_DIR, "Re_Write_questions.csv")
    if not os.path.exists(rw_path):
        for alt in [os.path.join(SCRIPT_DIR, "..", "Re_Write_questions.csv")]:
            if os.path.exists(alt):
                rw_path = alt
                break
    with open(rw_path, "r", encoding="utf-8-sig") as f:
        rw_questions = list(csv.DictReader(f))
    print(f"  ✅ Re_Write_questions.csv：{len(rw_questions)} 題")

    # ── 2. 切塊 ──
    print(f"\n📦 步驟 2：滑動視窗切塊（size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    print("-" * 40)
    chunks = chunk_texts(data_files)
    print(f"\n  📊 總計：{len(chunks)} 個切塊")

    # ── 3. 連接 API & Qdrant ──
    print(f"\n🔗 步驟 3：連接 Embedding API、LLM API、Qdrant")
    print("-" * 40)

    _, dim = get_embedding(["測試"])
    if dim is None:
        print("❌ Embedding API 不可用")
        return
    print(f"  ✅ Embedding API：維度 {dim}")

    test_llm = call_llm("你好", "回覆OK", temperature=0.1)
    if test_llm:
        print(f"  ✅ LLM API：{LLM_MODEL}")
    else:
        print("❌ LLM API 不可用")
        return

    client = QdrantClient(url="http://localhost:6333")
    print("  ✅ Qdrant 連接成功")

    # ── 4. 嵌入到 Qdrant（Dense） ──
    print(f"\n📤 步驟 4：嵌入到 Qdrant（Dense Vectors）")
    print("-" * 40)

    # 刪除舊 collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    all_points = []
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i + 50]
        embs, _ = get_embedding([c["text"] for c in batch])
        if embs is None:
            continue
        for j, (chunk, emb) in enumerate(zip(batch, embs)):
            all_points.append(PointStruct(
                id=i + j, vector=emb,
                payload={"text": chunk["text"], "source": chunk["source"]},
            ))
        time.sleep(0.2)

    client.upsert(collection_name=COLLECTION_NAME, points=all_points)
    print(f"  ✅ Qdrant：{len(all_points)} 個向量（Dense）")

    # ── 5. 建立 BM25 索引（Sparse） ──
    print(f"\n📚 步驟 5：建立 BM25 索引（Sparse / 關鍵字搜尋）")
    print("-" * 40)
    bm25_index = BM25Index(chunks)

    # ── 6. 處理 questions.csv（直接問題） ──
    print(f"\n🔍 步驟 6：處理 questions.csv（{len(questions)} 題）")
    print(f"  流程：Hybrid Search → ReRank → LLM Answer")
    print("-" * 40)

    q_results = []
    for q in questions:
        q_id = q["題目_ID"]
        q_text = q["題目"]
        print(f"\n  Q{q_id}: {q_text[:50]}...")

        answer, source = rag_pipeline(
            q_text, client, bm25_index, chunks, label=f"Q{q_id}"
        )

        q_results.append({
            "題目_ID": q_id,
            "題目": q_text,
            "標準答案": answer,
            "來源文件": source,
        })
        time.sleep(0.3)

    # 輸出 questions_answer.csv
    qa_path = os.path.join(SCRIPT_DIR, "questions_answer.csv")
    with open(qa_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["題目_ID", "題目", "標準答案", "來源文件"])
        writer.writeheader()
        writer.writerows(q_results)
    print(f"\n  ✅ questions_answer.csv：{len(q_results)} 筆")

    # ── 7. 處理 Re_Write_questions.csv（多輪對話） ──
    print(f"\n🔄 步驟 7：處理 Re_Write_questions.csv（{len(rw_questions)} 題）")
    print(f"  流程：Query ReWrite → Hybrid Search → ReRank → LLM Answer")
    print("-" * 40)

    # 按 conversation_id 分組
    conversations = {}
    for rw in rw_questions:
        cid = rw["conversation_id"]
        if cid not in conversations:
            conversations[cid] = []
        conversations[cid].append(rw)

    rw_results = []

    for cid, conv_questions in conversations.items():
        print(f"\n  💬 對話 {cid}（{len(conv_questions)} 輪）")
        history = []  # 累積對話歷史

        for rw in conv_questions:
            qid = rw["questions_id"]
            q_text = rw["questions"]
            print(f"    Q{cid}-{qid}: {q_text}")

            # 是否需要 Query ReWrite
            if len(history) > 0:
                # 有歷史 → 重寫查詢
                rewritten = rewrite_query(prompt_rewrite, history, q_text)
                print(f"      🔄 ReWrite: {rewritten[:60]}...")
                search_query = rewritten
            else:
                # 第一輪 → 直接搜尋
                search_query = q_text

            # RAG Pipeline
            answer, source = rag_pipeline(
                search_query, client, bm25_index, chunks,
                label=f"C{cid}-Q{qid}"
            )

            rw_results.append({
                "conversation_id": cid,
                "questions_id": qid,
                "questions": q_text,
                "answer": answer,
                "source": source,
            })

            # 累積歷史
            history.append({"role": "user", "content": q_text})
            history.append({"role": "assistant", "content": answer})

            time.sleep(0.3)

    # 輸出 Re_Write_answer.csv
    rwa_path = os.path.join(SCRIPT_DIR, "Re_Write_answer.csv")
    with open(rwa_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f,
                                fieldnames=["conversation_id", "questions_id",
                                            "questions", "answer", "source"])
        writer.writeheader()
        writer.writerows(rw_results)
    print(f"\n  ✅ Re_Write_answer.csv：{len(rw_results)} 筆")

    # ── 8. 輸出結果摘要 ──
    print(f"""
{'=' * 65}
✅ 課堂作業-04 完成！
{'=' * 65}

📋 系統架構：
  切塊方法：滑動視窗（size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）
  切塊數量：{len(chunks)} 塊
  Dense Search：Qdrant VDB（向量維度 {dim}）
  Sparse Search：BM25（jieba 中文分詞）
  Hybrid Fusion：Reciprocal Rank Fusion（k={RRF_K}）
  ReRank：LLM-based Relevance Scoring（{LLM_MODEL}）
  Answer Gen：LLM Top-{RERANK_TOP_K} → 精準萃取

📋 處理結果：
  questions_answer.csv：{len(q_results)} 題 → {qa_path}
  Re_Write_answer.csv：{len(rw_results)} 題 → {rwa_path}

📋 API 使用：
  Embedding：{EMBED_API_URL}
  LLM：{LLM_API_URL}（{LLM_MODEL}）
""")

    # 顯示答案預覽
    print("📝 questions_answer 預覽：")
    for r in q_results:
        ans_preview = r["標準答案"][:60] + "..." if len(r["標準答案"]) > 60 else r["標準答案"]
        print(f"  Q{r['題目_ID']}: {ans_preview} [{r['來源文件']}]")

    print(f"\n📝 Re_Write_answer 預覽：")
    for r in rw_results:
        ans_preview = r["answer"][:60] + "..." if len(r["answer"]) > 60 else r["answer"]
        print(f"  C{r['conversation_id']}-Q{r['questions_id']}: {ans_preview} [{r['source']}]")


if __name__ == "__main__":
    main()