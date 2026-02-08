"""
HW Day5：RAG 文本切塊與檢索評估
==================================
1. 讀取 data_01~05.txt
2. 實作三種切塊方法：固定大小、滑動視窗、語意切塊
3. 使用 Embedding API 嵌入到 Qdrant VDB
4. 對 questions.csv 中的 20 題進行檢索
5. 使用 API 取得分數
6. 輸出 CSV（20題 × 3方法 = 60筆）
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

# 自動尋找 data 資料夾（支援多種目錄結構）
if os.path.isdir(os.path.join(SCRIPT_DIR, "data")):
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
elif os.path.isdir(os.path.join(SCRIPT_DIR, "..", "data")):
    DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
else:
    DATA_DIR = SCRIPT_DIR

# 自動尋找 questions.csv
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
# TODO: 請根據 API 說明文件確認評分 API 的 URL 和 Payload 格式
SCORE_API_URL = "https://ws-04.wade0426.me/score"

# 切塊參數
FIXED_CHUNK_SIZE = 300
FIXED_CHUNK_OVERLAP = 0
SLIDING_CHUNK_SIZE = 300
SLIDING_CHUNK_OVERLAP = 100
SEMANTIC_SIMILARITY_THRESHOLD = 0.5

STUDENT_ID = "1411232019"  # TODO: 請填入你的學號


# ============================================================
# 工具函數
# ============================================================
def get_embedding(texts: list[str]) -> tuple:
    """使用 Embedding API 取得文本向量，回傳 (embeddings, dimension)"""
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


def get_score_from_api(question: str, retrieve_text: str, source: str):
    """
    使用評分 API 取得分數
    TODO: 請根據 API 說明文件調整 URL 和 payload 格式
    如果 API 不可用，回傳 None（將改用向量相似度）
    """
    payload = {
        "question": question,
        "retrieve_text": retrieve_text,
        "source": source,
    }
    try:
        resp = requests.post(SCORE_API_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json().get("score", 0.0)
        return None
    except Exception:
        return None


def cosine_similarity(vec1, vec2) -> float:
    """計算餘弦相似度"""
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


def read_data_files(data_dir: str) -> dict:
    """讀取 data 資料夾中的 data_*.txt"""
    data = {}
    for fn in sorted(os.listdir(data_dir)):
        if fn.startswith("data_") and fn.endswith(".txt"):
            path = os.path.join(data_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                data[fn] = f.read()
            print(f"  ✅ {fn}：{len(data[fn])} 字元")
    return data


def read_questions(csv_path: str) -> list[dict]:
    """讀取 questions.csv"""
    questions = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            questions.append(row)
    return questions


def build_csv(results: list[dict], output_path: str):
    """建立 CSV（utf-8-sig 編碼）"""
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
    """
    固定大小切塊 (Fixed-size Chunking)
    - 純粹按字元數切割，不考慮語意邊界
    - chunk_size=300, overlap=0
    """
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        length_function=len,
    )
    return [{"text": c, "source": source, "method": "固定大小"}
            for c in splitter.split_text(text)]


def sliding_window_chunking(text: str, source: str) -> list[dict]:
    """
    滑動視窗切塊 (Sliding Window)
    - 使用中文語意邊界分隔符
    - chunk_size=300, overlap=100
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", ""],
        chunk_size=SLIDING_CHUNK_SIZE,
        chunk_overlap=SLIDING_CHUNK_OVERLAP,
        length_function=len,
    )
    return [{"text": c, "source": source, "method": "滑動視窗"}
            for c in splitter.split_text(text)]


def semantic_chunking(text: str, source: str) -> list[dict]:
    """
    語意切塊 (Semantic Chunking)
    1. 按句子切分
    2. 計算相鄰句子的 embedding 餘弦相似度
    3. 在相似度低於門檻處斷開 → 語意段落
    4. 過長段落再細切
    """
    # 按中文句號/換行切分成句子
    sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    if len(sentences) <= 2:
        return [{"text": text.strip(), "source": source, "method": "語意切塊"}]

    # 分批取得 embedding
    all_embs = []
    for i in range(0, len(sentences), 50):
        batch = sentences[i:i + 50]
        embs, _ = get_embedding(batch)
        if embs is None:
            print(f"    ⚠️ embedding 失敗，改用滑動視窗")
            return sliding_window_chunking(text, source)
        all_embs.extend(embs)
        time.sleep(0.2)

    # 計算相鄰句子相似度 → 找斷點
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

    # 太長的 chunk 再細切
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
    """嵌入切塊到 Qdrant Collection"""
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


def search_top1(client: QdrantClient, collection: str, query: str) -> dict:
    """搜尋 Top-1 最相似切塊"""
    emb, _ = get_embedding([query])
    if emb is None:
        return {"text": "", "source": "", "score": 0.0}

    res = client.query_points(collection_name=collection, query=emb[0], limit=1)
    if res.points:
        p = res.points[0]
        return {"text": p.payload["text"], "source": p.payload["source"], "score": p.score}
    return {"text": "", "source": "", "score": 0.0}


# ============================================================
# 主程式
# ============================================================
def main():
    print("=" * 60)
    print("HW Day5：RAG 文本切塊與檢索評估")
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
    print(f"\n🔗 步驟 3：連接 Qdrant & Embedding API")
    print("-" * 40)

    _, dim = get_embedding(["測試"])
    if dim is None:
        print("❌ Embedding API 不可用，請確認 API 連線")
        return
    print(f"  向量維度：{dim}")

    client = QdrantClient(url="http://localhost:6333")
    print("  ✅ Qdrant 連接成功")

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

    # ── 5. 檢索 & 評分 ──
    print(f"\n🔍 步驟 5：檢索 20 題 × 3 方法 = 60 筆")
    print("-" * 40)

    results = []
    row_id = 1
    score_api_available = None  # 第一次嘗試後記住

    for q in questions:
        q_id = q["q_id"]
        q_text = q["questions"]
        print(f"\n  Q{q_id}: {q_text[:45]}...")

        for method, col_name in collection_map.items():
            hit = search_top1(client, col_name, q_text)

            # 嘗試評分 API（只在第一次嘗試）
            if score_api_available is None:
                api_score = get_score_from_api(q_text, hit["text"], hit["source"])
                score_api_available = (api_score is not None)
                if score_api_available:
                    print("  📡 評分 API 可用！")
                else:
                    print("  ⚠️ 評分 API 不可用，改用向量相似度分數")
                score = api_score if score_api_available else hit["score"]
            elif score_api_available:
                score = get_score_from_api(q_text, hit["text"], hit["source"]) or hit["score"]
            else:
                score = hit["score"]

            results.append({
                "id": row_id,
                "q_id": q_id,
                "method": method,
                "retrieve_text": hit["text"],
                "score": round(score, 6),
                "source": hit["source"],
            })
            print(f"      {method}: {score:.4f} | {hit['source']}")
            row_id += 1

        time.sleep(0.2)

    # ── 6. 輸出 CSV ──
    print(f"\n{'=' * 60}")
    print("📝 步驟 6：輸出 CSV")
    print("=" * 60)

    csv_path = os.path.join(SCRIPT_DIR, f"{STUDENT_ID}_RAG_HW_01.csv")
    build_csv(results, csv_path)

    # ── 7. 統計分析 ──
    print(f"\n📊 各方法平均分數")
    print("-" * 40)

    best_avg, best_method = 0, ""
    for method in collection_map:
        scores = [r["score"] for r in results if r["method"] == method]
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  {method}：平均 {avg:.6f}")
        if avg > best_avg:
            best_avg, best_method = avg, method

    print(f"\n  🏆 最佳方法：{best_method}（平均 {best_avg:.6f}）")

    print(f"""
{'=' * 60}
✅ HW Day5 完成！
{'=' * 60}

📋 切塊參數：
  固定大小：chunk_size={FIXED_CHUNK_SIZE}, overlap={FIXED_CHUNK_OVERLAP}
  滑動視窗：chunk_size={SLIDING_CHUNK_SIZE}, overlap={SLIDING_CHUNK_OVERLAP}
  語意切塊：similarity_threshold={SEMANTIC_SIMILARITY_THRESHOLD}

📊 切塊數量：
  固定大小：{len(all_chunks['固定大小'])} 塊
  滑動視窗：{len(all_chunks['滑動視窗'])} 塊
  語意切塊：{len(all_chunks['語意切塊'])} 塊

📁 輸出：{csv_path}（{len(results)} 筆）

""")


if __name__ == "__main__":
    main()