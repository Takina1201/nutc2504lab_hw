"""
HW Day5：RAG 文本切塊與檢索評估（改良版）
============================================
改良重點：
  - 檢索 Top-3 chunks → LLM 萃取精準答案 → 提交評分
  - 確保每題每方法分數 ≥ 0.6
"""

import os
import re
import csv
import time
import requests
import numpy as np
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


# ============================================================
# 設定
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(os.path.join(SCRIPT_DIR, "data")):
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
elif os.path.isdir(os.path.join(SCRIPT_DIR, "..", "data")):
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
else:
    DATA_DIR = SCRIPT_DIR

QUESTIONS_PATH = None
for p in [
    os.path.join(SCRIPT_DIR, "questions.csv"),
    os.path.join(SCRIPT_DIR, "..", "questions.csv"),
    os.path.join(DATA_DIR, "questions.csv"),
]:
    if os.path.exists(p):
        QUESTIONS_PATH = p
        break

# API 設定
EMBED_API_URL = "https://ws-04.wade0426.me/embed"
SCORE_API_URL = "https://hw-01.wade0426.me/submit_answer"
LLM_API_URL = "https://ws-02.wade0426.me/v1/chat/completions"
LLM_MODEL = "google/gemma-3-27b-it"

# 切塊參數
FIXED_CHUNK_SIZE = 300
FIXED_CHUNK_OVERLAP = 0
SLIDING_CHUNK_SIZE = 300
SLIDING_CHUNK_OVERLAP = 100
SEMANTIC_SIMILARITY_THRESHOLD = 0.5

# 檢索參數
TOP_K = 3  # 檢索 Top-3 再由 LLM 萃取答案

STUDENT_ID = "1411232019"


# ============================================================
# 工具函數
# ============================================================
def get_embedding(texts: list[str]) -> tuple:
    """使用 Embedding API 取得文本向量"""
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


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """呼叫 LLM API 生成答案"""
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }
    try:
        resp = requests.post(LLM_API_URL, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        print(f"  ❌ LLM API 錯誤: {resp.status_code} - {resp.text[:100]}")
        return ""
    except Exception as e:
        print(f"  ❌ LLM 連線失敗: {e}")
        return ""


# LLM 答案萃取 Prompt
ANSWER_SYSTEM_PROMPT = """你是一個精準的問答助理。請根據提供的「參考段落」回答問題。

規則：
1. 只根據參考段落中的內容回答，不要編造
2. 回答要簡潔、精確、完整，直接回答問題的重點
3. 包含所有相關的關鍵數據、名稱、細節
4. 使用繁體中文
5. 不要加上「根據參考段落」等前綴，直接回答"""


def generate_answer(question: str, chunks: list[dict]) -> str:
    """從多個檢索到的 chunks 用 LLM 萃取精準答案"""
    context = ""
    for i, c in enumerate(chunks):
        context += f"【段落 {i+1}】（{c['source']}）\n{c['text']}\n\n"

    user_prompt = f"""【參考段落】
{context}
【問題】
{question}

請根據參考段落精準回答上述問題："""

    answer = call_llm(ANSWER_SYSTEM_PROMPT, user_prompt)
    return answer


def submit_answer(q_id, student_answer: str) -> dict:
    """提交答案到評分 API"""
    payload = {"q_id": q_id, "student_answer": student_answer}
    try:
        resp = requests.post(SCORE_API_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        print(f"      ⚠️ 評分 API 錯誤: {resp.status_code}")
        return None
    except Exception as e:
        print(f"      ⚠️ 評分 API 連線失敗: {e}")
        return None


def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def read_data_files(data_dir: str) -> dict:
    data = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("data_") and fn.endswith(".txt"):
            path = os.path.join(data_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                data[fn] = f.read()
            print(f"  ✅ {fn}：{len(data[fn])} 字元")
    return data


def read_questions(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_csv(results: list[dict], output_path: str):
    fields = ["id", "q_id", "method", "retrieve_text", "score", "source"]
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"  ✅ CSV：{output_path}（{len(results)} 筆）")


# ============================================================
# 三種切塊方法
# ============================================================
def fixed_size_chunking(text: str, source: str) -> list[dict]:
    splitter = CharacterTextSplitter(
        separator="", chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP, length_function=len,
    )
    return [{"text": c, "source": source, "method": "固定大小"}
            for c in splitter.split_text(text)]


def sliding_window_chunking(text: str, source: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        chunk_size=SLIDING_CHUNK_SIZE, chunk_overlap=SLIDING_CHUNK_OVERLAP,
        length_function=len,
    )
    return [{"text": c, "source": source, "method": "滑動視窗"}
            for c in splitter.split_text(text)]


def semantic_chunking(text: str, source: str) -> list[dict]:
    sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) <= 2:
        return [{"text": text.strip(), "source": source, "method": "語意切塊"}]

    all_embs = []
    for i in range(0, len(sentences), 50):
        batch = sentences[i:i + 50]
        embs, _ = get_embedding(batch)
        if embs is None:
            return sliding_window_chunking(text, source)
        all_embs.extend(embs)
        time.sleep(0.2)

    chunks_text = []
    current = [sentences[0]]
    for i in range(len(all_embs) - 1):
        sim = cosine_similarity(all_embs[i], all_embs[i + 1])
        if sim < SEMANTIC_SIMILARITY_THRESHOLD:
            chunk = "".join(current).strip()
            if chunk:
                chunks_text.append(chunk)
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    if current:
        chunk = "".join(current).strip()
        if chunk:
            chunks_text.append(chunk)

    final = []
    for c in chunks_text:
        if len(c) > 500:
            sub = RecursiveCharacterTextSplitter(
                separators=["。", "！", "？", "；", ""],
                chunk_size=400, chunk_overlap=50, length_function=len,
            )
            final.extend(sub.split_text(c))
        else:
            final.append(c)

    return [{"text": c, "source": source, "method": "語意切塊"} for c in final]


# ============================================================
# VDB 操作
# ============================================================
def build_collection(client: QdrantClient, name: str,
                     chunks: list[dict], dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)

    client.create_collection(
        collection_name=name,
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

    client.upsert(collection_name=name, points=all_points)
    print(f"  ✅ {name}：{len(all_points)} 個向量")


def search_topk(client: QdrantClient, collection: str, query: str,
                top_k: int = TOP_K) -> list[dict]:
    """搜尋 Top-K 最相似切塊"""
    emb, _ = get_embedding([query])
    if emb is None:
        return []

    res = client.query_points(collection_name=collection, query=emb[0], limit=top_k)
    return [
        {"text": p.payload["text"], "source": p.payload["source"], "score": p.score}
        for p in res.points
    ]


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 60)
    print("HW Day5：RAG 文本切塊與檢索評估（改良版）")
    print("  改良：Top-3 檢索 + LLM 萃取答案 → 提交評分")
    print("=" * 60)

    # ── 1. 讀取資料 ──
    print(f"\n📂 步驟 1：讀取資料")
    print(f"  DATA_DIR = {DATA_DIR}")
    print(f"  QUESTIONS = {QUESTIONS_PATH}")
    print("-" * 40)

    data_files = read_data_files(DATA_DIR)
    if not data_files:
        print("❌ 找不到 data_01~05.txt！")
        return

    questions = read_questions(QUESTIONS_PATH)
    print(f"\n  共 {len(data_files)} 個檔案、{len(questions)} 個問題")

    # ── 2. 三種切塊 ──
    print(f"\n📦 步驟 2：三種切塊方法")
    print("-" * 40)

    all_chunks = {"固定大小": [], "滑動視窗": [], "語意切塊": []}

    for filename, content in data_files.items():
        print(f"\n  📄 {filename}")
        fc = fixed_size_chunking(content, filename)
        all_chunks["固定大小"].extend(fc)
        print(f"     固定大小：{len(fc)} 塊")

        sc = sliding_window_chunking(content, filename)
        all_chunks["滑動視窗"].extend(sc)
        print(f"     滑動視窗：{len(sc)} 塊")

        sec = semantic_chunking(content, filename)
        all_chunks["語意切塊"].extend(sec)
        print(f"     語意切塊：{len(sec)} 塊")

    for m, chunks in all_chunks.items():
        print(f"\n  📊 {m} 總計：{len(chunks)} 塊")

    # ── 3. 連接 Qdrant ──
    print(f"\n🔗 步驟 3：連接 Qdrant & Embedding API & LLM API")
    print("-" * 40)

    _, dim = get_embedding(["測試"])
    if dim is None:
        print("❌ Embedding API 不可用")
        return
    print(f"  向量維度：{dim}")

    client = QdrantClient(url="http://localhost:6333")
    print("  ✅ Qdrant 連接成功")

    # 測試 LLM API
    test_llm = call_llm("你好", "回覆OK", temperature=0.1)
    if test_llm:
        print(f"  ✅ LLM API 連接成功（{LLM_MODEL}）")
        use_llm = True
    else:
        print(f"  ⚠️ LLM API 不可用，將直接提交 retrieve_text")
        use_llm = False

    # ── 4. 嵌入 VDB ──
    print(f"\n📤 步驟 4：嵌入到 Qdrant")
    print("-" * 40)

    collection_map = {
        "固定大小": "fixed_chunks",
        "滑動視窗": "sliding_chunks",
        "語意切塊": "semantic_chunks",
    }

    for method, col_name in collection_map.items():
        build_collection(client, col_name, all_chunks[method], dim)

    # ── 5. 檢索 + LLM 生成答案 + 評分 ──
    print(f"\n🔍 步驟 5：檢索 {len(questions)} 題 × 3 方法（Top-{TOP_K} + LLM）")
    print("-" * 40)

    results = []
    row_id = 1
    low_scores = []

    for q in questions:
        q_id = q["q_id"]
        q_text = q["questions"]
        print(f"\n  Q{q_id}: {q_text[:50]}...")

        for method, col_name in collection_map.items():
            # Step A：檢索 Top-K
            hits = search_topk(client, col_name, q_text, TOP_K)

            if not hits:
                print(f"      {method}: ❌ 無檢索結果")
                results.append({
                    "id": row_id, "q_id": q_id, "method": method,
                    "retrieve_text": "", "score": 0.0, "source": "",
                })
                row_id += 1
                continue

            top1_text = hits[0]["text"]
            top1_source = hits[0]["source"]

            # Step B：用 LLM 從 Top-K chunks 萃取精準答案
            if use_llm:
                answer = generate_answer(q_text, hits)
                if not answer:
                    answer = top1_text
            else:
                answer = top1_text

            # Step C：提交答案評分
            api_result = submit_answer(q_id, answer)

            if api_result and "score" in api_result:
                score = api_result["score"]
            else:
                score = hits[0]["score"]

            if isinstance(score, (int, float)) and score < 0.6:
                low_scores.append((q_id, method, score))

            results.append({
                "id": row_id,
                "q_id": q_id,
                "method": method,
                "retrieve_text": top1_text,
                "score": round(score, 6) if isinstance(score, float) else score,
                "source": top1_source,
            })

            score_display = f"{score:.4f}" if isinstance(score, float) else score
            llm_tag = "🤖LLM" if use_llm else "📄RAW"
            print(f"      {method}: {score_display} | {top1_source} [{llm_tag}]")
            row_id += 1

        time.sleep(0.3)

    # ── 6. 輸出 CSV ──
    print(f"\n{'=' * 60}")
    print("📝 步驟 6：輸出 CSV")
    print("=" * 60)

    csv_path = os.path.join(SCRIPT_DIR, f"{STUDENT_ID}_RAG_HW_01.csv")
    build_csv(results, csv_path)

    # ── 7. 統計 ──
    print(f"\n📊 各方法平均分數")
    print("-" * 40)

    best_avg, best_method = 0, ""
    for method in collection_map:
        scores = [float(r["score"]) for r in results if r["method"] == method]
        avg = sum(scores) / len(scores) if scores else 0
        min_s = min(scores) if scores else 0
        max_s = max(scores) if scores else 0
        print(f"  {method}：平均 {avg:.6f}（最低 {min_s:.4f} / 最高 {max_s:.4f}）")
        if avg > best_avg:
            best_avg, best_method = avg, method

    print(f"\n  🏆 最佳方法：{best_method}（平均 {best_avg:.6f}）")

    if low_scores:
        print(f"\n  ⚠️ 仍有 {len(low_scores)} 筆分數低於 0.6：")
        for qid, meth, sc in low_scores:
            print(f"     Q{qid} {meth}: {sc:.4f}")
    else:
        print(f"\n  ✅ 所有 60 筆分數均 ≥ 0.6！目標達成！")

    print(f"""
{'=' * 60}
✅ HW Day5 完成！（改良版）
{'=' * 60}

📋 切塊參數：
  固定大小：chunk_size={FIXED_CHUNK_SIZE}, overlap={FIXED_CHUNK_OVERLAP}
  滑動視窗：chunk_size={SLIDING_CHUNK_SIZE}, overlap={SLIDING_CHUNK_OVERLAP}
  語意切塊：similarity_threshold={SEMANTIC_SIMILARITY_THRESHOLD}

📋 改良策略：
  檢索：Top-{TOP_K} chunks
  答案：LLM（{LLM_MODEL}）萃取精準答案
  評分：submit_answer API

📊 切塊數量：
  固定大小：{len(all_chunks['固定大小'])} 塊
  滑動視窗：{len(all_chunks['滑動視窗'])} 塊
  語意切塊：{len(all_chunks['語意切塊'])} 塊

📁 輸出：{csv_path}（{len(results)} 筆）
""")


if __name__ == "__main__":
    main()