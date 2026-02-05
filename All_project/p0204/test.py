"""
LangChain 進階討論 - 自定義工具 (Custom Tools) 完整實作
包含：
1. 基本工具定義
2. JSON Schema 展示
3. 直接工具呼叫
4. 與 LLM 整合（含 Tool Calling）
5. 簡易 Agent 循環

根據投影片內容：ch4-1 實作
API: https://ws-02.wade0426.me/v1
Model: google/gemma-3-27b-it
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json

# ============================================================
# Part 1: 定義自訂工具 (Custom Tool)
# ============================================================

@tool
def extract_order_data(name: str, phone: str, product: str, quantity: int, address: str):
    """
    資料提取專用工具。
    專門用於從非結構化文本中提取訂單相關資訊（姓名、電話、商品、數量、地址）。
    """
    return {
        "name": name,
        "phone": phone,
        "product": product,
        "quantity": quantity,
        "address": address
    }


@tool
def calculate_total_price(product: str, quantity: int, unit_price: float) -> dict:
    """
    計算訂單總價的工具。
    根據商品名稱、數量和單價計算總金額。
    """
    total = quantity * unit_price
    return {
        "product": product,
        "quantity": quantity,
        "unit_price": unit_price,
        "total_price": total
    }


@tool
def validate_phone_number(phone: str) -> dict:
    """
    驗證電話號碼格式的工具。
    檢查電話號碼是否為有效的台灣手機號碼格式（09開頭，共10碼）。
    """
    cleaned = phone.replace(" ", "").replace("-", "")
    is_valid = (
        len(cleaned) == 10 and 
        cleaned.startswith("09") and 
        cleaned.isdigit()
    )
    return {
        "original": phone,
        "cleaned": cleaned,
        "is_valid": is_valid,
        "message": "有效的手機號碼" if is_valid else "無效的手機號碼格式"
    }


# ============================================================
# Part 2: 查看工具的 JSON Schema
# ============================================================

def show_tool_schema():
    """展示工具被轉換成的 JSON Schema（這是 LLM 真正看到的格式）"""
    print("=" * 60)
    print("📋 LLM 真正看到的工具定義 (JSON Schema)")
    print("=" * 60)
    
    tools = [extract_order_data, calculate_total_price, validate_phone_number]
    
    for t in tools:
        print(f"\n📦 工具名稱: {t.name}")
        print(f"📝 描述: {t.description}")
        print(f"🔧 參數 Schema:")
        # 使用新版 Pydantic 的方法
        schema = t.args_schema.model_json_schema()
        print(json.dumps(schema, indent=2, ensure_ascii=False))
        print("-" * 40)


# ============================================================
# Part 3: 直接呼叫工具（不透過 LLM）
# ============================================================

def demo_direct_tool_call():
    """示範直接呼叫工具"""
    print("\n" + "=" * 60)
    print("🔧 直接呼叫工具測試（不需要 LLM）")
    print("=" * 60)
    
    # 方法 1: 使用 .invoke() 傳入 dict
    print("\n--- 使用 .invoke() ---")
    result = extract_order_data.invoke({
        "name": "王小明",
        "phone": "0912345678",
        "product": "藍牙耳機",
        "quantity": 3,
        "address": "台北市信義區"
    })
    print(f"extract_order_data 結果:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    
    # 電話驗證測試
    print("\n--- 電話驗證測試 ---")
    test_phones = ["0912-345-678", "0912345678", "02-1234-5678"]
    for phone in test_phones:
        result = validate_phone_number.invoke({"phone": phone})
        status = "✅" if result["is_valid"] else "❌"
        print(f"{status} {phone:15} -> {result['message']}")


# ============================================================
# Part 4: 與 LLM 整合 - Tool Calling
# ============================================================

def create_llm_with_tools():
    """建立綁定了工具的 LLM"""
    llm = ChatOpenAI(
        base_url="https://ws-02.wade0426.me/v1",
        api_key="not-needed",
        model="google/gemma-3-27b-it",
        temperature=0
    )
    
    tools = [extract_order_data, calculate_total_price, validate_phone_number]
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools, {t.name: t for t in tools}


def demo_tool_calling():
    """示範 LLM Tool Calling 流程"""
    print("\n" + "=" * 60)
    print("🤖 LLM Tool Calling 流程示範")
    print("=" * 60)
    
    llm_with_tools, tools_dict = create_llm_with_tools()
    
    # 準備訊息
    user_input = "我叫王小明，電話 0912-345-678，想要訂購 3 個藍牙耳機，請寄到台北市信義區松仁路 100 號"
    
    print(f"\n📝 用戶輸入: {user_input}")
    
    messages = [
        {"role": "system", "content": "你是訂單處理助手。請使用工具提取訂單資訊。"},
        {"role": "user", "content": user_input}
    ]
    
    # Step 1: 呼叫 LLM
    print("\n🔄 Step 1: 呼叫 LLM...")
    response = llm_with_tools.invoke(messages)
    
    # Step 2: 檢查是否有 Tool Calls
    if response.tool_calls:
        print(f"✅ LLM 決定呼叫 {len(response.tool_calls)} 個工具")
        
        messages.append(response)
        
        # Step 3: 執行每個工具
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\n🔧 Tool Call #{i}:")
            print(f"   工具名稱: {tool_call['name']}")
            print(f"   參數: {json.dumps(tool_call['args'], ensure_ascii=False)}")
            
            # 執行工具
            tool_result = tools_dict[tool_call['name']].invoke(tool_call['args'])
            print(f"   結果: {json.dumps(tool_result, ensure_ascii=False)}")
            
            # 將結果加入對話
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_result, ensure_ascii=False),
                    tool_call_id=tool_call["id"]
                )
            )
        
        # Step 4: 取得最終回應
        print("\n🔄 Step 4: 取得最終回應...")
        final_response = llm_with_tools.invoke(messages)
        print(f"\n📋 最終回應:\n{final_response.content}")
    else:
        print("ℹ️ LLM 沒有呼叫任何工具")
        print(f"📋 回應: {response.content}")


# ============================================================
# Part 5: 完整 Agent 循環（可重複使用的函數）
# ============================================================

def run_order_agent(user_input: str, verbose: bool = True) -> str:
    """
    執行訂單處理 Agent
    
    Args:
        user_input: 用戶的自然語言輸入
        verbose: 是否顯示詳細過程
    
    Returns:
        Agent 的最終回應
    """
    llm_with_tools, tools_dict = create_llm_with_tools()
    
    system_prompt = """你是一個專業的訂單處理助手。
你的任務是從用戶的自然語言描述中提取訂單資訊。

工具使用指南：
- extract_order_data: 提取訂單的基本資訊
- calculate_total_price: 如果有價格資訊，計算總價
- validate_phone_number: 驗證電話號碼格式

請用繁體中文回應，並清楚整理提取到的資訊。"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Agent 循環（最多 3 輪）
    for iteration in range(3):
        response = llm_with_tools.invoke(messages)
        
        if not response.tool_calls:
            # 沒有工具呼叫，返回回應
            return response.content
        
        if verbose:
            print(f"\n🔄 迭代 {iteration + 1}: 執行 {len(response.tool_calls)} 個工具呼叫")
        
        messages.append(response)
        
        for tool_call in response.tool_calls:
            if verbose:
                print(f"   🔧 {tool_call['name']}: {tool_call['args']}")
            
            if tool_call['name'] in tools_dict:
                result = tools_dict[tool_call['name']].invoke(tool_call['args'])
                messages.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_call["id"]
                    )
                )
    
    # 最後一次呼叫取得最終回應
    final_response = llm_with_tools.invoke(messages)
    return final_response.content


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("🚀 LangChain 自定義工具完整教學")
    print("=" * 60)
    
    # Part 1 & 2: 展示工具定義和 Schema
    show_tool_schema()
    
    # Part 3: 直接呼叫工具
    demo_direct_tool_call()
    
    # Part 4: LLM Tool Calling（需要 API 連線）
    print("\n" + "=" * 60)
    print("⚠️  以下測試需要連接到 API")
    print("=" * 60)
    
    try:
        demo_tool_calling()
    except Exception as e:
        print(f"❌ API 連線錯誤: {e}")
        print("💡 如果 API 無法連線，可以先研究上面的工具定義和直接呼叫範例")
    
    print("\n" + "=" * 60)
    print("✅ Demo 完成！")
    print("=" * 60)