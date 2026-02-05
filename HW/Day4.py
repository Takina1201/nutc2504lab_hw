"""
課後實戰 — 自動查證 AI v2.0 (全面升級版)
========================================

升級內容：
1. 持久化快取 + TTL 過期機制
2. 重試機制 + 更完善的錯誤處理
3. 多來源整合閱讀（不只讀第一筆）
4. 豐富的 State 設計（來源追蹤、搜尋歷史、信心度）
5. 更細緻的 Planner 決策（JSON 評估）
6. 非同步併發處理（同時讀取多個網頁）
7. 搜尋去重（避免重複搜尋相同關鍵字）

安裝需求：
pip install langgraph langchain langchain-openai playwright requests aiohttp
python -m playwright install-deps
python -m playwright install chromium

API 配置：
- LLM: https://ws-03.wade0426.me/v1 (gpt-oss-120b)
- SearXNG: https://puli-8080.huannago.com/search
"""

import os
import time
import json
import base64
import hashlib
import asyncio
import requests
from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# Playwright 非同步版本
from playwright.async_api import async_playwright, Browser, Page

# LangChain / LangGraph 相關
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

# ============================================================
# 1. 設定與初始化
# ============================================================

API_KEY = ""
BASE_URL = "https://ws-03.wade0426.me/v1"
MODEL_NAME = "/models/gpt-oss-120b"
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# 快取設定
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 86400  # 快取有效期：24 小時

# 搜尋設定
MAX_SEARCH_STEPS = 3
MAX_RETRIES = 3
VLM_READ_COUNT = 2  # 每次搜尋讀取前幾筆結果

# 初始化共用 LLM
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.1
)


# ============================================================
# 2. 持久化快取系統
# ============================================================

@dataclass
class CacheEntry:
    """快取項目結構"""
    question: str
    answer: str
    sources: List[str]
    timestamp: float
    confidence: float
    
    def is_expired(self, ttl: int = CACHE_TTL) -> bool:
        """檢查是否過期"""
        return time.time() - self.timestamp > ttl
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(**data)


class PersistentCache:
    """持久化快取管理器"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
        # 記憶體快取（加速讀取）
        self._memory_cache: Dict[str, CacheEntry] = {}
    
    def _get_cache_key(self, question: str) -> str:
        """生成快取 key（使用 MD5 hash）"""
        # 正規化問題（去除多餘空白）
        normalized = " ".join(question.strip().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """取得快取檔案路徑"""
        return self.cache_dir / f"{key}.json"
    
    def get(self, question: str) -> Optional[CacheEntry]:
        """取得快取"""
        key = self._get_cache_key(question)
        
        # 先檢查記憶體快取
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired(self.ttl):
                return entry
            else:
                # 過期就刪除
                del self._memory_cache[key]
        
        # 再檢查檔案快取
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding='utf-8'))
                entry = CacheEntry.from_dict(data)
                if not entry.is_expired(self.ttl):
                    # 載入到記憶體快取
                    self._memory_cache[key] = entry
                    return entry
                else:
                    # 過期就刪除檔案
                    cache_path.unlink()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"    快取讀取失敗: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, question: str, answer: str, sources: List[str], confidence: float = 0.8):
        """儲存快取"""
        key = self._get_cache_key(question)
        entry = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            timestamp=time.time(),
            confidence=confidence
        )
        
        # 儲存到記憶體
        self._memory_cache[key] = entry
        
        # 儲存到檔案
        cache_path = self._get_cache_path(key)
        cache_path.write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    
    def clear_expired(self):
        """清理過期快取"""
        cleared = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                data = json.loads(cache_file.read_text(encoding='utf-8'))
                entry = CacheEntry.from_dict(data)
                if entry.is_expired(self.ttl):
                    cache_file.unlink()
                    cleared += 1
            except:
                cache_file.unlink()
                cleared += 1
        return cleared
    
    def get_stats(self) -> dict:
        """取得快取統計"""
        total = len(list(self.cache_dir.glob("*.json")))
        memory_count = len(self._memory_cache)
        return {
            "total_cached": total,
            "in_memory": memory_count,
            "cache_dir": str(self.cache_dir)
        }


# 初始化快取
cache = PersistentCache()


# ============================================================
# 3. 搜尋工具（含重試機制）
# ============================================================

def search_searxng(query: str, limit: int = 3, retries: int = MAX_RETRIES) -> List[dict]:
    """
    [工具] 執行 SearXNG 搜尋（含重試機制）
    """
    print(f"    [SearXNG] 正在搜尋: {query}")
    
    params = {
        "q": query,
        "format": "json",
        "language": "zh-TW"
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(SEARXNG_URL, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # 過濾有 URL 的結果，並去除重複
                results = []
                seen_urls = set()
                
                for r in data.get('results', []):
                    url = r.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            'title': r.get('title', '無標題'),
                            'url': url,
                            'content': r.get('content', ''),
                            'engine': r.get('engine', 'unknown')
                        })
                
                print(f"    [SearXNG] 找到 {len(results)} 筆結果")
                return results[:limit]
            
            elif response.status_code == 429:
                # Rate limit，等待後重試
                wait_time = 2 ** attempt
                print(f"    [SearXNG] Rate limited，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"    [SearXNG] HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"    [SearXNG] 逾時，重試 {attempt + 1}/{retries}")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"    [SearXNG] 連線錯誤: {e}")
            time.sleep(1)
    
    return []


# ============================================================
# 4. VLM 視覺閱讀（非同步 + 重試）
# ============================================================

async def vlm_read_single_page(
    browser: Browser, 
    url: str, 
    title: str,
    max_screenshots: int = 2
) -> dict:
    """
    [非同步] 讀取單一網頁
    """
    result = {
        "url": url,
        "title": title,
        "content": "",
        "success": False,
        "error": None
    }
    
    context = None
    try:
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 1200},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        # 設定請求攔截（阻擋廣告和追蹤器）
        await page.route("**/*", lambda route: (
            route.abort() if any(x in route.request.url for x in [
                'analytics', 'tracking', 'ads', 'doubleclick', 'facebook.com/tr'
            ]) else route.continue_()
        ))
        
        # 前往網頁
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        await page.wait_for_timeout(2000)
        
        # 注入 CSS 隱藏干擾元素
        await page.add_style_tag(content="""
            iframe, .ad, .ads, .advertisement, 
            [class*="cookie"], [class*="popup"], 
            [class*="modal"], [class*="overlay"] {
                opacity: 0 !important;
                pointer-events: none !important;
                display: none !important;
            }
        """)
        
        # 滾動截圖
        screenshots_b64 = []
        for i in range(max_screenshots):
            screenshot = await page.screenshot(type='png')
            b64 = base64.b64encode(screenshot).decode('utf-8')
            screenshots_b64.append(b64)
            await page.evaluate("window.scrollBy(0, 800)")
            await page.wait_for_timeout(800)
        
        # 使用 VLM 分析
        if screenshots_b64:
            content = await analyze_screenshots_with_vlm(screenshots_b64, title)
            result["content"] = content
            result["success"] = True
            
    except Exception as e:
        result["error"] = str(e)
        print(f"    [VLM] 讀取失敗 ({title}): {e}")
    finally:
        if context:
            await context.close()
    
    return result


async def analyze_screenshots_with_vlm(screenshots_b64: List[str], title: str) -> str:
    """
    [VLM] 分析截圖內容
    """
    msg_content = [
        {
            "type": "text",
            "text": f"""這是網頁「{title}」的截圖。
請閱讀並提取：
1. 主要內容和關鍵資訊
2. 重要數據或統計
3. 新聞重點或結論
忽略廣告、選單和無關內容。用繁體中文回答。"""
        }
    ]
    
    for img in screenshots_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    
    try:
        # LangChain 的 invoke 是同步的，這裡用 run_in_executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke([HumanMessage(content=msg_content)])
        )
        return response.content
    except Exception as e:
        return f"VLM 分析失敗: {e}"


async def vlm_read_websites_parallel(
    urls: List[dict],
    max_concurrent: int = 3
) -> List[dict]:
    """
    [非同步] 並行讀取多個網頁
    """
    print(f"    [VLM] 啟動並行閱讀 {len(urls)} 個網頁...")
    
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox"
            ]
        )
        
        # 使用信號量控制並行數
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def read_with_semaphore(url_info):
            async with semaphore:
                return await vlm_read_single_page(
                    browser,
                    url_info['url'],
                    url_info['title']
                )
        
        tasks = [read_with_semaphore(url_info) for url_info in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        await browser.close()
    
    # 處理結果
    processed_results = []
    for r in results:
        if isinstance(r, Exception):
            processed_results.append({
                "url": "",
                "title": "",
                "content": f"讀取失敗: {r}",
                "success": False
            })
        else:
            processed_results.append(r)
    
    success_count = sum(1 for r in processed_results if r.get('success'))
    print(f"    [VLM] 完成，成功 {success_count}/{len(urls)}")
    
    return processed_results


def synthesize_sources(sources: List[dict], question: str) -> str:
    """
    [LLM] 整合多來源資訊
    """
    if not sources:
        return "無法取得任何資訊。"
    
    # 建立來源摘要
    source_texts = []
    for i, s in enumerate(sources):
        if s.get('success') and s.get('content'):
            source_texts.append(f"""
【來源 {i+1}: {s['title']}】
URL: {s['url']}
內容:
{s['content']}
""")
    
    if not source_texts:
        return "所有網頁讀取失敗，請稍後再試。"
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個專業的資訊整合專家。
請根據以下多個來源的資訊，針對使用者問題進行整合分析。

使用者問題：{question}

來源資訊：
{sources}

請：
1. 整合各來源的關鍵資訊
2. 標註資訊來自哪個來源
3. 如有矛盾，指出不同說法
4. 用繁體中文回答""")
    ])
    
    try:
        result = (prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "sources": "\n".join(source_texts)
        })
        return result
    except Exception as e:
        # 如果整合失敗，直接返回原始內容
        return "\n\n".join(source_texts)


def execute_search_tool(query: str, question: str, search_history: List[str]) -> dict:
    """
    [整合工具] 搜尋 + VLM 閱讀 + 來源整合
    """
    # 檢查是否重複搜尋
    if query in search_history:
        print(f"    [搜尋] 跳過重複關鍵字: {query}")
        return {
            "content": f"已搜尋過「{query}」，跳過重複搜尋。",
            "sources": [],
            "is_duplicate": True
        }
    
    # 1. 執行搜尋
    results = search_searxng(query, limit=5)
    
    if not results:
        return {
            "content": "未找到相關結果，請嘗試其他關鍵字。",
            "sources": [],
            "is_duplicate": False
        }
    
    # 2. 準備要讀取的 URL
    urls_to_read = results[:VLM_READ_COUNT]
    
    # 3. 並行 VLM 閱讀
    vlm_results = asyncio.run(vlm_read_websites_parallel(urls_to_read))
    
    # 4. 整合來源
    synthesized = synthesize_sources(vlm_results, question)
    
    # 5. 收集成功的來源 URL
    source_urls = [r['url'] for r in vlm_results if r.get('success')]
    
    return {
        "content": synthesized,
        "sources": source_urls,
        "is_duplicate": False,
        "search_results_summary": [
            {"title": r['title'], "url": r['url']} 
            for r in results
        ]
    }


# ============================================================
# 5. 定義 State（豐富版）
# ============================================================

class AgentState(TypedDict):
    # 基本欄位
    question: str               # 原始問題
    knowledge_base: str         # 累積的資訊
    search_query: str           # 當前生成的關鍵字
    steps: int                  # 搜尋步數
    final_answer: str           # 最終答案
    is_sufficient: bool         # 決策結果
    
    # 新增欄位
    sources: List[str]          # 所有來源 URL
    search_history: List[str]   # 搜尋歷史（避免重複）
    confidence: float           # 答案信心度 (0-1)
    planner_analysis: dict      # Planner 的詳細分析


# ============================================================
# 6. 定義 Nodes
# ============================================================

def check_cache_node(state: AgentState) -> dict:
    """[Node 1] 快取檢查"""
    question = state["question"]
    print(f"\n[1] 快取檢查: {question}")
    
    cached = cache.get(question)
    
    if cached:
        print(f"    ✓ 命中快取！(信心度: {cached.confidence:.0%})")
        return {
            "final_answer": cached.answer,
            "sources": cached.sources,
            "confidence": cached.confidence,
            "is_sufficient": True
        }
    else:
        print("    ✗ 無快取，進入決策流程。")
        return {
            "knowledge_base": "",
            "steps": 0,
            "sources": [],
            "search_history": [],
            "confidence": 0.0,
            "is_sufficient": False
        }


def planner_node(state: AgentState) -> dict:
    """[Node 2] 決策（細緻版，使用 JSON 評估）"""
    print(f"[2] AI 決策 (Planner)... (步數: {state['steps']}/{MAX_SEARCH_STEPS})")
    
    # 安全機制：最多搜尋次數
    if state["steps"] >= MAX_SEARCH_STEPS:
        print("    ⚠ 已達最大搜尋次數，強制回答。")
        return {
            "is_sufficient": True,
            "planner_analysis": {
                "completeness": 5,
                "credibility": 5,
                "need_more_search": False,
                "reason": "達到最大搜尋次數限制"
            }
        }
    
    # 第一次搜尋前，直接判定需要搜尋
    if state["steps"] == 0 and not state.get("knowledge_base"):
        print("    → 首次查詢，需要搜尋資訊。")
        return {
            "is_sufficient": False,
            "planner_analysis": {
                "completeness": 0,
                "credibility": 0,
                "need_more_search": True,
                "reason": "尚未收集任何資訊"
            }
        }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個嚴謹的查證決策者。請評估目前收集的資訊是否足以回答問題。

使用者問題：{question}

目前收集到的資訊：
---
{context}
---

已搜尋過的關鍵字：{search_history}

請用 JSON 格式回應（不要加 markdown）：
{{
    "completeness": <1-10 資訊完整度>,
    "credibility": <1-10 來源可信度>,
    "need_more_search": <true/false>,
    "reason": "<簡短說明理由>",
    "suggested_query": "<如果需要更多搜尋，建議的關鍵字>"
}}

評估標準：
- completeness >= 7 且 credibility >= 6 才算足夠
- 如果已有足夠資訊但缺乏佐證，可再搜尋一次驗證""")
    ])
    
    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "question": state["question"],
            "context": state.get("knowledge_base", "（無資訊）"),
            "search_history": ", ".join(state.get("search_history", [])) or "（無）"
        })
        
        # 解析 JSON
        # 移除可能的 markdown 標記
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        response = response.strip()
        
        analysis = json.loads(response)
        
        completeness = analysis.get("completeness", 0)
        credibility = analysis.get("credibility", 0)
        need_more = analysis.get("need_more_search", True)
        
        is_sufficient = (completeness >= 7 and credibility >= 6) or not need_more
        confidence = (completeness + credibility) / 20  # 轉換為 0-1
        
        print(f"    完整度: {completeness}/10, 可信度: {credibility}/10")
        print(f"    決策: {'資訊足夠' if is_sufficient else '需要更多搜尋'}")
        if analysis.get("reason"):
            print(f"    理由: {analysis['reason']}")
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence,
            "planner_analysis": analysis,
            # 如果 Planner 有建議關鍵字，先存起來
            "search_query": analysis.get("suggested_query", "")
        }
        
    except json.JSONDecodeError as e:
        print(f"    ⚠ JSON 解析失敗，使用保守策略")
        # 保守策略：繼續搜尋
        return {
            "is_sufficient": state["steps"] >= 2,  # 至少搜尋兩次
            "confidence": 0.5,
            "planner_analysis": {"error": str(e)}
        }


def query_gen_node(state: AgentState) -> dict:
    """[Node 3] 生成搜尋關鍵字"""
    print("[3] 生成關鍵字 (Query Gen)...")
    
    # 如果 Planner 已經建議了關鍵字，直接使用
    if state.get("search_query"):
        suggested = state["search_query"]
        if suggested not in state.get("search_history", []):
            print(f"    使用 Planner 建議: {suggested}")
            return {"search_query": suggested}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個搜尋專家。根據問題與已知資訊，生成最適合的搜尋關鍵字。

問題：{question}

已知資訊：
{context}

已搜尋過（避免重複）：{search_history}

規則：
1. 只輸出一個搜尋關鍵字或短語
2. 不要加引號或其他符號
3. 使用繁體中文
4. 避免與已搜尋過的關鍵字重複或過於相似
5. 如果需要驗證資訊，可以搜尋相關的權威來源

輸出關鍵字：""")
    ])
    
    query = (prompt | llm | StrOutputParser()).invoke({
        "question": state["question"],
        "context": state.get("knowledge_base", "（無）")[:500],
        "search_history": ", ".join(state.get("search_history", [])) or "（無）"
    }).strip()
    
    # 清理輸出
    query = query.replace('"', '').replace("'", "").strip()
    
    print(f"    生成: {query}")
    return {"search_query": query}


def search_tool_node(state: AgentState) -> dict:
    """[Node 4] 執行搜尋與 VLM 閱讀"""
    print("[4] 執行搜尋與視覺閱讀...")
    
    query = state["search_query"]
    search_history = state.get("search_history", [])
    
    # 執行整合工具
    result = execute_search_tool(query, state["question"], search_history)
    
    # 更新狀態
    new_history = search_history + [query]
    new_sources = list(set(state.get("sources", []) + result.get("sources", [])))
    
    # 更新知識庫
    if not result.get("is_duplicate"):
        new_kb = state.get("knowledge_base", "")
        new_kb += f"\n\n=== 搜尋「{query}」的結果 ===\n{result['content']}"
    else:
        new_kb = state.get("knowledge_base", "")
    
    return {
        "knowledge_base": new_kb,
        "search_history": new_history,
        "sources": new_sources,
        "steps": state["steps"] + 1
    }


def final_answer_node(state: AgentState) -> dict:
    """[Node 5] 產生最終報告"""
    print("[5] 整理最終報告...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個專業的查證報告撰寫者。請根據收集到的資訊回答使用者問題。

問題：{question}

收集到的資訊：
{context}

參考來源：{sources}

請：
1. 用繁體中文完整回答問題
2. 整理關鍵發現和重點
3. 如有不確定或矛盾的資訊，請說明
4. 在回答末尾列出主要參考來源

回答：""")
    ])
    
    sources_text = "\n".join([f"- {url}" for url in state.get("sources", [])]) or "無外部來源"
    
    answer = (prompt | llm | StrOutputParser()).invoke({
        "question": state["question"],
        "context": state.get("knowledge_base", "（無資訊）"),
        "sources": sources_text
    })
    
    # 計算最終信心度
    confidence = state.get("confidence", 0.5)
    if state.get("planner_analysis"):
        analysis = state["planner_analysis"]
        if "completeness" in analysis and "credibility" in analysis:
            confidence = (analysis["completeness"] + analysis["credibility"]) / 20
    
    # 寫入快取
    cache.set(
        question=state["question"],
        answer=answer,
        sources=state.get("sources", []),
        confidence=confidence
    )
    
    print(f"    ✓ 已存入快取 (信心度: {confidence:.0%})")
    
    return {
        "final_answer": answer,
        "confidence": confidence
    }


# ============================================================
# 7. 建立 Graph
# ============================================================

workflow = StateGraph(AgentState)

# 加入節點
workflow.add_node("check_cache", check_cache_node)
workflow.add_node("planner", planner_node)
workflow.add_node("query_gen", query_gen_node)
workflow.add_node("search_tool", search_tool_node)
workflow.add_node("final_answer", final_answer_node)

# 設定進入點
workflow.set_entry_point("check_cache")


# 條件邊函式
def route_cache(state: AgentState) -> str:
    """快取路由"""
    if state.get("final_answer"):
        return "end"
    return "planner"


def route_planner(state: AgentState) -> str:
    """決策路由"""
    if state.get("is_sufficient"):
        return "final_answer"
    return "query_gen"


# 設定條件邊
workflow.add_conditional_edges(
    "check_cache",
    route_cache,
    {"end": END, "planner": "planner"}
)

workflow.add_conditional_edges(
    "planner",
    route_planner,
    {"final_answer": "final_answer", "query_gen": "query_gen"}
)

# 普通邊
workflow.add_edge("query_gen", "search_tool")
workflow.add_edge("search_tool", "planner")
workflow.add_edge("final_answer", END)

# 編譯
app = workflow.compile()


# ============================================================
# 8. 輔助函式
# ============================================================

def print_banner():
    """印出啟動橫幅"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║         自動查證 AI v1.1 (功能擴充版)                      ║
╠═══════════════════════════════════════════════════════════╣
║  功能：                                                    ║
║  ✓ 持久化快取 + TTL 過期機制                               ║
║  ✓ 重試機制 + 完善錯誤處理                                 ║
║  ✓ 多來源並行 VLM 閱讀                                     ║
║  ✓ 細緻的 Planner 決策 (JSON 評估)                         ║
║  ✓ 搜尋去重 + 來源追蹤                                     ║
╠═══════════════════════════════════════════════════════════╣
║  指令：                                                    ║
║  q / exit    - 離開程式                                    ║
║  /cache      - 查看快取統計                                ║
║  /clear      - 清理過期快取                                ║
║  /graph      - 顯示流程圖                                  ║
╚═══════════════════════════════════════════════════════════╝
""")


def handle_command(cmd: str) -> bool:
    """處理特殊指令，回傳是否為指令"""
    cmd = cmd.strip().lower()
    
    if cmd == "/cache":
        stats = cache.get_stats()
        print(f"\n📊 快取統計：")
        print(f"   總快取數：{stats['total_cached']}")
        print(f"   記憶體中：{stats['in_memory']}")
        print(f"   快取目錄：{stats['cache_dir']}")
        return True
    
    elif cmd == "/clear":
        cleared = cache.clear_expired()
        print(f"\n🗑️  已清理 {cleared} 筆過期快取")
        return True
    
    elif cmd == "/graph":
        print("\n📊 流程圖：")
        print(app.get_graph().draw_ascii())
        return True
    
    return False


# ============================================================
# 9. 主程式
# ============================================================

if __name__ == "__main__":
    print_banner()
    
    # 啟動時清理過期快取
    cleared = cache.clear_expired()
    if cleared:
        print(f"🗑️  啟動時清理了 {cleared} 筆過期快取\n")
    
    print("-" * 60)
    
    while True:
        try:
            q = input("\n請輸入問題: ").strip()
            
            if not q:
                continue
            
            if q.lower() in ["q", "exit", "quit"]:
                print("\n👋 再見！")
                break
            
            # 檢查是否為指令
            if q.startswith("/"):
                if handle_command(q):
                    continue
            
            # 執行查證
            print("\n" + "─" * 60)
            
            start_time = time.time()
            result = app.invoke({"question": q})
            elapsed = time.time() - start_time
            
            print("\n" + "═" * 60)
            print("📋 最終回答")
            print("═" * 60)
            print(result["final_answer"])
            print("─" * 60)
            print(f"⏱️  耗時: {elapsed:.1f} 秒")
            print(f"🎯 信心度: {result.get('confidence', 0):.0%}")
            if result.get("sources"):
                print(f"📚 來源數: {len(result['sources'])} 個")
            print("═" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 收到中斷信號，再見！")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            import traceback
            traceback.print_exc()