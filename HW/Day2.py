import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化模型
llm = ChatOpenAI(
    base_url="https://ws-02.wade0426.me/v1",
    api_key="vllm-token",
    model="google/gemma-3-27b-it",
    temperature=0
)

# 2. 定義兩個不同風格的 Prompt
# 分身 A: Instagram 網紅
prompt_ig = ChatPromptTemplate.from_template(
    "你是一個熱愛生活的 IG 網紅。請針對主題『{topic}』寫1句繁體中文貼文，要有很多 Emoji 🔥，語氣要超嗨，像是跟粉絲聊天。"
)

# 分身 B: LinkedIn 職場專家
prompt_linkedin = ChatPromptTemplate.from_template(
    "你是一個專業的企業顧問。請針對主題『{topic}』寫1句繁體中文的 LinkedIn 貼文，分析其對商業運作的啟示，語氣要專業、理性、簡潔。"
)

# 分身 C: professor 專業學者
prompt_professor = ChatPromptTemplate.from_template(
    "你是一個專業的學者顧問。請針對主題『{topic}』寫1句繁體中文的研究報告，研究其對物理學和自然界和科學界的奧妙，語氣要專業、理性、且可考性十足。"
)

# 3. 建立兩條支線 (Chain)
chain_ig = prompt_ig | llm | StrOutputParser()
chain_linkedin = prompt_linkedin | llm | StrOutputParser()
chain_professor = prompt_professor | llm | StrOutputParser()

# 4. 定義平行處理主線
combined_chain = RunnableParallel(
    instagram=chain_ig,
    linkedin=chain_linkedin,
    professor=chain_professor
)

# 【關鍵修改】: 改成讓使用者在終端機輸入
print("===" * 10)
topic = input("請輸入你想討論的主題 (例如: 天氣很冷、AI取代人類...): ")
print(f"\n🔥 確認主題: {topic}\n")

# --- 第一部分：流式輸出 (Streaming) ---
print("=== 測試 1: 流式輸出 (觀察平行運算) ===")
# 這裡加個 try-except 避免使用者直接按 Enter 沒輸入東西導致報錯
if topic.strip():
    for chunk in combined_chain.stream({"topic": topic}):
        print(chunk,flush=True)
else:
    print("❌ 你沒有輸入任何主題喔！")

print("\n" + "="*40 + "\n")

# --- 第二部分：批次處理 (Batch) 與計時 ---
if topic.strip():
    print("=== 測試 2: 批次處理 (計算總耗時) ===")
    start_time = time.time()

    # 執行 batch
    result_list = combined_chain.batch([{"topic": topic}])

    end_time = time.time()
    duration = end_time - start_time
    final_result = result_list[0]
    
    # 顯示最終成果
    print(f"🕒 耗時: {duration:.2f} 秒\n")

    print("【LinkedIn 專家說】:")
    print(final_result['linkedin'])
    print("\n" + "-"*20 + "\n")

    print("【IG 網紅說】:")
    print(final_result['instagram'])
    print("\n" + "-"*20 + "\n")
    
    print("【專業學者說】:")
    print(final_result['professor'])