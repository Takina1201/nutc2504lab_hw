"""
課堂作業-01：Qdrant 向量資料庫實作
1. 建立 Qdrant Collection 並連接
2. 建立五個 Point 或更多
3. 使用 API 獲得向量
4. 嵌入到 VDB
5. 召回內容
"""

import csv
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    SearchParams, QuantizationSearchParams
)

# ============================================================
# Embedding API
# ============================================================
EMBED_API_URL = "https://ws-04.wade0426.me/embed"


def get_embedding(texts: list[str]):
    """使用 API 取得文本的 embedding 向量，回傳 (embeddings, dimension)"""
    data = {
        "texts": texts,
        "normalize": True,
        "batch_size": 32
    }
    response = requests.post(EMBED_API_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        print(f"成功取得 embedding，維度: {result['dimension']}")
        return result["embeddings"], result["dimension"]
    else:
        print(f"錯誤: {response.status_code} - {response.text}")
        return None, None


# ============================================================
# 1. 連接 Qdrant 並建立 Collection
# ============================================================
client = QdrantClient(url="http://localhost:6333")
print("Qdrant 連接成功！")

# 動態取得向量維度（不寫死 4096）
_, VECTOR_SIZE = get_embedding(["測試"])
print(f"向量維度: {VECTOR_SIZE}")

COLLECTION_NAME = "news_collection"

# 如果 collection 已存在就先刪除（方便重複執行）
collections = client.get_collections().collections
if any(c.name == COLLECTION_NAME for c in collections):
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"已刪除舊的 collection: {COLLECTION_NAME}")

# 建立 Collection，size 由 API 動態取得
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print(f"Collection '{COLLECTION_NAME}' 建立成功！向量維度: {VECTOR_SIZE}")


# ============================================================
# 2. 準備資料（五筆以上）
# ============================================================
documents = [
    {
        "id": 1,
        "text": "伊朗自2025年12月底起爆發大規模罷工與示威抗爭，起因為失控的物價通膨與經濟危機。伊朗政府將抗爭定調為恐怖主義行為，全面切斷網路並動員武力鎮壓，截至2026年1月13日已超過2400人死亡。",
        "source": "data_01.txt",
        "topic": "國際政治",
        "year": 2026
    },
    {
        "id": 2,
        "text": "國家太空中心的影像處理中心負責將衛星回傳的數據解碼、校正、重組成可讀懂的地表影像。校正流程分為輻射校正與幾何校正，前者修正大氣或儀器偏差，後者校準地理坐標，使影像能用於防災與地理研究。",
        "source": "data_02.txt",
        "topic": "太空科技",
        "year": 2025
    },
    {
        "id": 3,
        "text": "台灣醫院急重症人力崩塌持續擴大，從20年前內外婦兒四大皆空，演變至今年內外婦兒急神經六大皆空。直美現象指醫學生畢業後直接進入醫美市場，加劇急重症科人力流失。2026年健保進入兆元時代。",
        "source": "data_03.txt",
        "topic": "醫療健保",
        "year": 2025
    },
    {
        "id": 4,
        "text": "二戰後至少8國執行超過2056次核試驗。放射性物質鍶-90半衰期達28年會在骨骼中積聚導致腫瘤，鈽-239半衰期長達2.44萬年，攝取1微克就會造成嚴重危害。美國在馬紹爾群島進行66次核試驗，釋放能量相當於7232顆廣島原子彈。",
        "source": "data_04.txt",
        "topic": "核武議題",
        "year": 2025
    },
    {
        "id": 5,
        "text": "福衛八號是台灣第一個自製衛星星系，計畫6+2顆衛星，84%關鍵元件由國內研製。其國安任務包括蒐集具戰略性價值的影像，高解析度光學遙測可監測中國海警船動態並偵測暗目標，預計2031年完成星系布建。",
        "source": "data_05.txt",
        "topic": "國防安全",
        "year": 2025
    },
    {
        "id": 6,
        "text": "伊朗經濟危機的三大原因：國際社會重新收緊的經濟制裁與石油禁運、2025年6月以色列與美國對伊朗發動的跨境空襲、以及極端氣候引發的60年來最嚴重缺水危機，德黑蘭11個月總降水量僅1.1毫米。",
        "source": "data_01.txt",
        "topic": "國際政治",
        "year": 2025
    },
    {
        "id": 7,
        "text": "台灣透過守望亞洲國際志願計畫參與國際救災合作。2023年協助菲律賓油輪漏油追蹤，2024年協助日本石川能登半島地震災害分析。台灣的衛星影像處理能力已成為實質的外交工具。",
        "source": "data_02.txt",
        "topic": "太空科技",
        "year": 2024
    },
    {
        "id": 8,
        "text": "OSINT公開來源情報是從公開或商業可用資訊中獲取的情報。溫約瑟利用Google Earth、哥白尼哨兵衛星等免費遙測影像製作出解放軍基地與設施地圖，透過影像堆疊可精細研析軍事動作與戰術。",
        "source": "data_05.txt",
        "topic": "國防安全",
        "year": 2025
    },
    {
        "id": 9,
        "text": "SIPRI報告指出截至2025年1月全球約有12241枚核彈頭，9614枚屬可供部署的軍事庫存。美俄最後一份新削減戰略武器條約將於2026年2月到期，9個擁核國都計畫擴增核武庫，全球已開啟新一輪核武威脅時代。",
        "source": "data_04.txt",
        "topic": "核武議題",
        "year": 2025
    },
    {
        "id": 10,
        "text": "衛福部長石崇良的首波改革方向對準醫療人力盤點與直美設限，企圖把健保集中火力投在最需要的醫療缺口上。健保兆元時代面臨超高齡人口、高科技和新藥價格飆漲的挑戰，規模已近中央總預算的三分之一。",
        "source": "data_03.txt",
        "topic": "醫療健保",
        "year": 2026
    }
]

print(f"共準備 {len(documents)} 筆資料")


# ============================================================
# 3. 使用 API 獲得向量 & 4. 嵌入到 VDB
# ============================================================
all_texts = [doc["text"] for doc in documents]
embeddings, _ = get_embedding(all_texts)

points = []
for doc, embedding in zip(documents, embeddings):
    points.append(
        PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={
                "text": doc["text"],
                "source": doc["source"],
                "topic": doc["topic"],
                "year": doc["year"]
            }
        )
    )

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)
print(f"成功插入 {len(points)} 筆資料到 '{COLLECTION_NAME}'！")

# 確認資料筆數
collection_info = client.get_collection(collection_name=COLLECTION_NAME)
print(f"Collection 中共有 {collection_info.points_count} 筆資料")
print(f"向量維度: {collection_info.config.params.vectors.size}")
print(f"距離度量: {collection_info.config.params.vectors.distance}")


# ============================================================
# 5. 召回內容（向量搜尋）
# ============================================================
def search_similar(query: str, limit: int = 3):
    """根據查詢文本搜尋最相似的資料"""
    query_embedding, _ = get_embedding([query])

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding[0],
        limit=limit
    )

    print(f"\n查詢: 「{query}」")
    print(f"找到 {len(search_result.points)} 筆相似結果：")
    print("=" * 60)

    for point in search_result.points:
        print(f"ID: {point.id}")
        print(f"相似度分數 (Score): {point.score:.4f}")
        print(f"主題: {point.payload['topic']}")
        print(f"來源: {point.payload['source']}")
        print(f"內容: {point.payload['text'][:100]}...")
        print("-" * 60)

    return search_result


# 測試搜尋
print("\n" + "=" * 60)
print("【搜尋方式 1】基本向量搜尋")
print("=" * 60)

search_similar("台灣的衛星有什麼功能？")
search_similar("台灣醫院缺人的問題有多嚴重？")
search_similar("核試驗對人類有什麼危害？")
search_similar("伊朗為什麼會爆發抗爭？")


# ============================================================
# 【搜尋方式 2】單一條件 Filter（精確匹配 topic）
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 2】單一條件 Filter - 只搜尋「國防安全」主題")
print("=" * 60)

query = "國家安全與軍事"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchValue(value="國防安全")
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 主題: {point.payload['topic']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 3】MatchAny - 同時搜尋多個主題
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 3】MatchAny - 同時搜尋「太空科技」和「國防安全」")
print("=" * 60)

query = "衛星對台灣的價值"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="topic",
                match=MatchAny(any=["太空科技", "國防安全"])
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 主題: {point.payload['topic']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 4】Range Filter - 依照年份範圍篩選
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 4】Range Filter - 只搜尋 2026 年的資料")
print("=" * 60)

query = "最新的政策與事件"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="year",
                range=Range(gte=2026)  # 大於等於 2026
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 年份: {point.payload['year']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 5】must_not - 排除特定主題
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 5】must_not - 搜尋時排除「國際政治」主題")
print("=" * 60)

query = "危機與威脅"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must_not=[
            FieldCondition(
                key="topic",
                match=MatchValue(value="國際政治")
            )
        ]
    ),
    limit=3
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 主題: {point.payload['topic']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 6】組合條件 - must + must_not + Range
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 6】組合條件 - 2025年以後 且 排除「核武議題」")
print("=" * 60)

query = "台灣的發展"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="year",
                range=Range(gte=2025)
            )
        ],
        must_not=[
            FieldCondition(
                key="topic",
                match=MatchValue(value="核武議題")
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 主題: {point.payload['topic']} | 年份: {point.payload['year']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 7】依 source 檔案篩選
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 7】依 source 篩選 - 只搜尋來自 data_01.txt 的資料")
print("=" * 60)

query = "中東局勢"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    query_filter=Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value="data_01.txt")
            )
        ]
    ),
    limit=5
)

for point in results.points:
    print(f"ID: {point.id} | Score: {point.score:.4f} | 來源: {point.payload['source']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 8】用 ID 直接取得特定 Point
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 8】用 ID 直接取得特定 Point")
print("=" * 60)

result = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1, 5, 9],
    with_vectors=False  # 不需要回傳向量，節省空間
)

for point in result:
    print(f"ID: {point.id} | 主題: {point.payload['topic']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 【搜尋方式 9】Scroll 分頁瀏覽全部資料
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 9】Scroll 分頁瀏覽（每頁 3 筆）")
print("=" * 60)

page = 1
offset = None

while True:
    results, next_offset = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        offset=offset,
        with_vectors=False
    )

    if not results:
        break

    print(f"\n--- 第 {page} 頁 ---")
    for point in results:
        print(f"  ID: {point.id} | 主題: {point.payload['topic']} | 來源: {point.payload['source']}")

    offset = next_offset
    page += 1

    if offset is None:
        break


# ============================================================
# 【搜尋方式 10】Count 統計符合條件的資料數量
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 10】Count 統計各主題的資料數量")
print("=" * 60)

topics = ["國際政治", "太空科技", "醫療健保", "核武議題", "國防安全"]

for topic in topics:
    count_result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic)
                )
            ]
        ),
        exact=True
    )
    print(f"  {topic}: {count_result.count} 筆")


# ============================================================
# 【搜尋方式 11】Score Threshold - 只回傳相似度超過門檻的結果
# ============================================================
print("\n" + "=" * 60)
print("【搜尋方式 11】Score Threshold - 只回傳相似度 > 0.5 的結果")
print("=" * 60)

query = "醫療人力不足的解決方案"
query_embedding, _ = get_embedding([query])

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding[0],
    score_threshold=0.5,  # 只回傳 score > 0.5
    limit=10
)

print(f"查詢: 「{query}」")
print(f"相似度 > 0.5 的結果共 {len(results.points)} 筆：")
for point in results.points:
    print(f"  ID: {point.id} | Score: {point.score:.4f} | 主題: {point.payload['topic']}")
    print(f"  內容: {point.payload['text'][:80]}...")
    print("-" * 60)


# ============================================================
# 使用 questions.csv 進行批次搜尋測試
# ============================================================
print("\n" + "=" * 60)
print("使用 questions.csv 批次搜尋")
print("=" * 60)

questions = []
with open(r"D:\nutc2504lab_hw\All_project\p0206\HW\questions.csv", "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    for row in reader:
        questions.append(row)

print(f"共載入 {len(questions)} 個問題")

# 取前 5 題做示範搜尋
for q in questions[:5]:
    search_similar(q["questions"], limit=2)