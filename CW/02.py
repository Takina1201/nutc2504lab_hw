"""
課堂作業-02：文本切塊與向量資料庫檢索比較
============================================
1. 下載範例檔案（text.txt）
2. 實作固定切塊
3. 實作滑動視窗切塊
4. 嵌入到 VDB (Qdrant)
5. 試著召回並比較兩種切塊方法
6. 試著處理表格（table 資料夾）
7. 上傳到 GitHub (CW/02)

嵌入模型：使用 TF-IDF + TruncatedSVD（LSA）產生密集向量
向量資料庫：Qdrant（記憶體模式）
"""

import os
import re
import numpy as np
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


# ============================================================
# 嵌入模型：TF-IDF + SVD (LSA) — 不需下載模型
# ============================================================
class LocalEmbedding:
    """
    使用 TF-IDF + TruncatedSVD（Latent Semantic Analysis）
    產生密集向量嵌入，無需下載預訓練模型。
    
    原理：
    1. TF-IDF 將文本轉為稀疏向量（詞頻-逆文件頻率）
    2. TruncatedSVD 降維到固定維度（類似 LSA）
    3. L2 正規化，使向量長度為 1（方便計算 cosine similarity）
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",  # 字元級分析，適合中文
            ngram_range=(1, 3),  # 1~3 字元的 n-gram
            max_features=5000,   # 最多 5000 個特徵
        )
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.is_fitted = False

    def fit(self, texts: list[str]):
        """用所有文本訓練嵌入模型"""
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        # SVD 維度不能超過 min(樣本數, 特徵數) - 1
        max_dim = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1]) - 1
        actual_dim = min(self.dim, max_dim)
        if actual_dim != self.dim:
            self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
            self.dim = actual_dim
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        print(f"✅ 嵌入模型訓練完成：維度={self.dim}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """將文本轉為密集向量"""
        if not self.is_fitted:
            raise ValueError("請先呼叫 fit() 訓練模型")
        tfidf_matrix = self.vectorizer.transform(texts)
        dense_vectors = self.svd.transform(tfidf_matrix)
        dense_vectors = normalize(dense_vectors, norm="l2")
        return dense_vectors


# ============================================================
# 步驟 1：讀取範例檔案 text.txt
# ============================================================
print("=" * 60)
print("步驟 1：讀取範例檔案")
print("=" * 60)

# 取得腳本所在目錄，確保路徑正確
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_PATH = os.path.join(SCRIPT_DIR, "text.txt")

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"原始文本長度：{len(raw_text)} 字元")
print(f"前 200 字：{raw_text[:200]}...")
print()


# ============================================================
# 步驟 2：實作固定切塊 (Fixed-size Chunking)
# ============================================================
print("=" * 60)
print("步驟 2：固定切塊 (Fixed-size Chunking)")
print("=" * 60)

fixed_splitter = CharacterTextSplitter(
    separator="",          # 不使用特定分隔符，純粹按字元數切割
    chunk_size=200,        # 每個 chunk 200 字元
    chunk_overlap=0,       # 固定切塊不重疊
    length_function=len,
)

fixed_chunks = fixed_splitter.split_text(raw_text)

print(f"固定切塊數量：{len(fixed_chunks)} 個\n")
for i, chunk in enumerate(fixed_chunks[:5]):
    print(f"--- 固定切塊 [{i+1}] (長度: {len(chunk)}) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    print()
print(f"...（共 {len(fixed_chunks)} 個切塊）\n")


# ============================================================
# 步驟 3：實作滑動視窗切塊 (Sliding Window Chunking)
# ============================================================
print("=" * 60)
print("步驟 3：滑動視窗切塊 (Sliding Window Chunking)")
print("=" * 60)

sliding_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。", "！", "？", "；", ""],
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
)

sliding_chunks = sliding_splitter.split_text(raw_text)

print(f"滑動視窗切塊數量：{len(sliding_chunks)} 個\n")
for i, chunk in enumerate(sliding_chunks[:5]):
    print(f"--- 滑動視窗切塊 [{i+1}] (長度: {len(chunk)}) ---")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)
    print()
print(f"...（共 {len(sliding_chunks)} 個切塊）\n")


# ============================================================
# 步驟 4：嵌入到 VDB (Qdrant)
# ============================================================
print("=" * 60)
print("步驟 4：嵌入到 Qdrant 向量資料庫")
print("=" * 60)

embedding_model = LocalEmbedding(dim=128)
all_texts = fixed_chunks + sliding_chunks
embedding_model.fit(all_texts)

qdrant_client = QdrantClient(":memory:")
embedding_dim = embedding_model.dim

# 固定切塊 Collection
qdrant_client.create_collection(
    collection_name="fixed_chunks",
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)
fixed_embeddings = embedding_model.embed(fixed_chunks)
fixed_points = [
    PointStruct(id=i, vector=emb.tolist(),
                payload={"text": chunk, "chunk_id": i, "method": "fixed"})
    for i, (chunk, emb) in enumerate(zip(fixed_chunks, fixed_embeddings))
]
qdrant_client.upsert(collection_name="fixed_chunks", points=fixed_points)
print(f"✅ 固定切塊已嵌入 Qdrant：{len(fixed_points)} 個向量")

# 滑動視窗切塊 Collection
qdrant_client.create_collection(
    collection_name="sliding_chunks",
    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
)
sliding_embeddings = embedding_model.embed(sliding_chunks)
sliding_points = [
    PointStruct(id=i, vector=emb.tolist(),
                payload={"text": chunk, "chunk_id": i, "method": "sliding"})
    for i, (chunk, emb) in enumerate(zip(sliding_chunks, sliding_embeddings))
]
qdrant_client.upsert(collection_name="sliding_chunks", points=sliding_points)
print(f"✅ 滑動視窗切塊已嵌入 Qdrant：{len(sliding_points)} 個向量\n")


# ============================================================
# 步驟 5：試著召回並比較兩種切塊方法
# ============================================================
print("=" * 60)
print("步驟 5：召回比較兩種切塊方法")
print("=" * 60)

test_queries = [
    "Graph RAG 有哪些檢索策略？",
    "微軟 GraphRAG 的核心特點是什麼？",
    "知識圖譜如何解決幻覺問題？",
]

for query in test_queries:
    print(f"\n🔍 查詢：{query}")
    print("-" * 50)

    query_vector = embedding_model.embed([query])[0].tolist()

    fixed_results = qdrant_client.query_points(
        collection_name="fixed_chunks", query=query_vector, limit=3).points

    print("\n📦 【固定切塊】Top-3 結果：")
    for rank, result in enumerate(fixed_results, 1):
        text_preview = result.payload["text"][:80].replace("\n", " ")
        print(f"  [{rank}] 分數: {result.score:.4f} | {text_preview}...")

    sliding_results = qdrant_client.query_points(
        collection_name="sliding_chunks", query=query_vector, limit=3).points

    print("\n🪟 【滑動視窗切塊】Top-3 結果：")
    for rank, result in enumerate(sliding_results, 1):
        text_preview = result.payload["text"][:80].replace("\n", " ")
        print(f"  [{rank}] 分數: {result.score:.4f} | {text_preview}...")

    fixed_best = fixed_results[0].score if fixed_results else 0
    sliding_best = sliding_results[0].score if sliding_results else 0
    winner = "滑動視窗" if sliding_best > fixed_best else "固定切塊"
    print(f"\n  ⭐ 最佳匹配：{winner}（固定: {fixed_best:.4f} vs 滑動: {sliding_best:.4f}）")


# 比較分析總結
print("\n" + "=" * 60)
print("比較分析總結")
print("=" * 60)
print(f"""
┌──────────────────┬──────────────────┬──────────────────────┐
│     比較項目       │    固定切塊       │    滑動視窗切塊       │
├──────────────────┼──────────────────┼──────────────────────┤
│   切塊數量         │    {len(fixed_chunks):>4} 個        │     {len(sliding_chunks):>4} 個           │
│   chunk_size      │    200 字元       │     200 字元          │
│   chunk_overlap   │    0 字元         │     50 字元           │
│   分隔符策略       │    無（純字數）    │     語意邊界          │
│   語意完整性       │    可能被截斷      │     盡量保持完整       │
│   資訊重疊         │    無             │     有重疊區域         │
│   適用場景         │    快速分割        │     高品質 RAG        │
└──────────────────┴──────────────────┴──────────────────────┘

重點差異：
1. 固定切塊可能在句子中間截斷，導致語意不完整
2. 滑動視窗透過 overlap 保留上下文，減少資訊遺失
3. 滑動視窗使用語意邊界（句號、問號等）切割，語意更完整
4. 滑動視窗的切塊數量通常較多（因為有重疊）
""")


# ============================================================
# 步驟 6：試著處理表格（table 資料夾）
# ============================================================
print("=" * 60)
print("步驟 6：處理表格資料（table 資料夾）")
print("=" * 60)

TABLE_DIR = os.path.join(SCRIPT_DIR, "table")

# ─── 方法 1：Markdown 表格（table_txt.md）───
print("\n📄 方法 1：Markdown 表格處理（table_txt.md）")
print("-" * 40)

md_path = os.path.join(TABLE_DIR, "table_txt.md")
with open(md_path, "r", encoding="utf-8") as f:
    table_md_content = f.read().strip()

print("原始 Markdown 表格：")
print(table_md_content[:300] + "..." if len(table_md_content) > 300 else table_md_content)

# 解析 Markdown 表格：逐列切塊，每列保留表頭
table_lines = table_md_content.strip().split("\n")
header = table_lines[0]
separator_line = table_lines[1] if len(table_lines) > 1 and set(table_lines[1].replace("|","").replace("-","").strip()) <= {""} else ""
data_start = 2 if separator_line else 1
data_rows = table_lines[data_start:]

table_chunks_md = []
for row in data_rows:
    if row.strip():
        chunk = f"{header}\n{separator_line}\n{row}" if separator_line else f"{header}\n{row}"
        table_chunks_md.append(chunk)

print(f"\n✅ Markdown 表格切塊數量：{len(table_chunks_md)} 個")
for i, row in enumerate(data_rows):
    cols = [c.strip() for c in row.split("|") if c.strip()]
    if cols:
        print(f"  切塊 [{i+1}] 項目: {cols[0]}")


# ─── 方法 2：HTML 表格（table_html.html）───
print("\n\n📄 方法 2：HTML 表格處理（table_html.html）")
print("-" * 40)

html_path = os.path.join(TABLE_DIR, "table_html.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

print(f"HTML 檔案大小：{len(html_content)} 字元")

# 從 HTML 中提取 <table> 區塊
table_match = re.search(r"<table.*?>(.*?)</table>", html_content, re.DOTALL)
if not table_match:
    print("⚠️ 未找到 <table> 標籤")
    table_chunks_html = []
else:
    table_html_content = table_match.group(0)

    # 提取所有 <tr> 列
    rows_html = re.findall(r"<tr.*?>(.*?)</tr>", table_html_content, re.DOTALL)

    # 找到表頭（<th> 標籤）
    html_headers = []
    header_row_idx = 0
    for idx, row in enumerate(rows_html):
        ths = re.findall(r"<th.*?>(.*?)</th>", row, re.DOTALL)
        if ths:
            # 清除 HTML 標籤
            html_headers = [re.sub(r"<.*?>", "", h).strip() for h in ths]
            header_row_idx = idx
            break

    print(f"表頭欄位：{html_headers}")

    # 提取每一列資料，轉換為自然語言描述
    table_chunks_html = []
    for row_html in rows_html[header_row_idx + 1:]:
        # 提取 <td> 中的內容，清除內部 HTML 標籤（如 <strong>、<br>）
        cells_raw = re.findall(r"<td.*?>(.*?)</td>", row_html, re.DOTALL)
        cells = [re.sub(r"<.*?>", "", c).strip() for c in cells_raw]

        if cells and len(cells) == len(html_headers):
            # 方式 A：結構化格式（表頭：內容）
            parts = [f"{h}：{c}" for h, c in zip(html_headers, cells)]
            chunk = "；".join(parts)
            table_chunks_html.append(chunk)

    print(f"✅ HTML 表格切塊數量：{len(table_chunks_html)} 個")
    for i, chunk in enumerate(table_chunks_html):
        preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
        print(f"  切塊 [{i+1}]：{preview}")


# ─── 表格切塊嵌入 VDB ───
print("\n\n📦 將表格切塊嵌入 Qdrant")

all_table_chunks = table_chunks_md + table_chunks_html
embedding_model_table = LocalEmbedding(dim=64)
embedding_model_table.fit(all_texts + all_table_chunks)

qdrant_client.create_collection(
    collection_name="table_chunks",
    vectors_config=VectorParams(size=embedding_model_table.dim, distance=Distance.COSINE),
)

table_embeddings = embedding_model_table.embed(all_table_chunks)
table_points = [
    PointStruct(id=i, vector=emb.tolist(),
                payload={"text": chunk, "chunk_id": i,
                         "source": "markdown" if i < len(table_chunks_md) else "html"})
    for i, (chunk, emb) in enumerate(zip(all_table_chunks, table_embeddings))
]
qdrant_client.upsert(collection_name="table_chunks", points=table_points)
print(f"✅ 表格切塊已嵌入 Qdrant：{len(table_points)} 個向量")

# 表格查詢測試
table_queries = [
    "三民校區的重點發展計畫是什麼？",
    "哪個校區跟航太有關？",
]

for query in table_queries:
    print(f"\n🔍 表格查詢：{query}")
    query_vector = embedding_model_table.embed([query])[0].tolist()
    table_results = qdrant_client.query_points(
        collection_name="table_chunks", query=query_vector, limit=3).points

    print("📊 Top-3 結果：")
    for rank, result in enumerate(table_results, 1):
        src = result.payload["source"]
        text = result.payload["text"][:100]
        print(f"  [{rank}] 分數: {result.score:.4f} | 來源: {src} | {text}...")

print("""
📝 表格處理方法總結：
  1. Markdown 表格 → 逐列切塊（每列保留表頭）
  2. HTML 表格 → 解析後轉自然語言描述
  兩種方式都確保每個切塊包含完整的「項目-值」對應關係。
""")


# ============================================================
# 完成！
# ============================================================
print("=" * 60)
print("✅ 課堂作業-02 全部完成！")
print("=" * 60)
print(f"""
完成項目：
  1. ✅ 讀取範例檔案 text.txt（{len(raw_text)} 字元）
  2. ✅ 實作固定切塊（CharacterTextSplitter, size=200, overlap=0 → {len(fixed_chunks)} 塊）
  3. ✅ 實作滑動視窗切塊（RecursiveCharacterTextSplitter, size=200, overlap=50 → {len(sliding_chunks)} 塊）
  4. ✅ 嵌入到 Qdrant VDB（TF-IDF + SVD 嵌入，dim={embedding_dim}）
  5. ✅ 召回比較兩種切塊方法（3 個測試查詢）
  6. ✅ 處理表格（Markdown {len(table_chunks_md)} 塊 + HTML {len(table_chunks_html)} 塊）
""")