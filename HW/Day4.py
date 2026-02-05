"""
自動查證 AI v1.1 (精準查證版)
========================================

全面升級內容：
1. 來源可信度評分系統 — 區分官方、新聞、維基、論壇等
2. 交叉驗證機制 — 要求多個獨立來源確認事實
3. 時效性檢查 — 優先使用較新資訊，標註資訊日期
4. 矛盾偵測與解決 — 發現矛盾時自動深入搜尋
5. 事實聲明提取 — 將資訊拆解為可驗證的聲明
6. 多角度搜尋策略 — 自動生成不同角度的搜尋關鍵字
7. 結構化驗證報告 — 清晰呈現每個事實的驗證狀態

安裝需求：
pip install langgraph langchain langchain-openai playwright requests aiohttp
python -m playwright install-deps
python -m playwright install chromium

API 配置：
- LLM: https://ws-03.wade0426.me/v1 (gpt-oss-120b)
- SearXNG: https://puli-8080.huannago.com/search

課後練習規定：
- 必須使用優化方式：✓ 快取機制
- 必要節點：✓ planner, query_gen, search_tool
"""

import os
import re
import time
import json
import base64
import hashlib
import asyncio
import requests
from typing import TypedDict, List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from urllib.parse import urlparse
from enum import Enum

# Playwright 非同步版本
from playwright.async_api import async_playwright, Browser, Page

# LangChain / LangGraph 相關
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# ============================================================
# 1. 設定與初始化
# ============================================================

API_KEY = ""
BASE_URL = "https://ws-03.wade0426.me/v1"
MODEL_NAME = "/models/gpt-oss-120b"
SEARXNG_URL = "https://puli-8080.huannago.com/search"

# 快取設定（優化方式 1：快取機制）
CACHE_DIR = Path("./cache_v3")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 43200  # 快取有效期：12 小時

# 搜尋設定
MAX_SEARCH_ROUNDS = 4      # 最大搜尋輪數
MAX_RETRIES = 3            # API 重試次數
VLM_READ_COUNT = 3         # 每輪讀取網頁數
MIN_SOURCES_FOR_CONFIDENCE = 2  # 最少需要幾個來源才有信心

# 初始化共用 LLM
llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.1
)

# 用於複雜推理的 LLM（溫度稍高）
llm_reasoning = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=MODEL_NAME,
    temperature=0.3
)


# ============================================================
# 2. 來源可信度評分系統
# ============================================================

class SourceCredibility(Enum):
    """來源可信度等級"""
    OFFICIAL = 5        # 官方來源（政府、官方網站）
    ACADEMIC = 5        # 學術來源（.edu、學術期刊）
    MAJOR_NEWS = 4      # 主流媒體（BBC、Reuters、聯合報等）
    WIKIPEDIA = 3       # 維基百科（需交叉驗證）
    TECH_MEDIA = 3      # 科技媒體（TechCrunch、數位時代等）
    LOCAL_NEWS = 3      # 地方新聞
    BLOG = 2            # 部落格、Medium
    FORUM = 1           # 論壇、PTT、Dcard
    CONTENT_FARM = 0    # 內容農場
    UNKNOWN = 1         # 未知來源


# 可信來源網域對照表
CREDIBILITY_DOMAINS = {
    # 官方來源
    SourceCredibility.OFFICIAL: [
        'gov.tw', 'gov.cn', 'gov.uk', 'gov', 'edu.tw', 'edu',
        'who.int', 'un.org', 'nasa.gov', 'nih.gov',
        'apple.com', 'google.com', 'microsoft.com', 'meta.com',
        'tesla.com', 'openai.com', 'anthropic.com',
    ],
    # 學術來源
    SourceCredibility.ACADEMIC: [
        'nature.com', 'science.org', 'ieee.org', 'acm.org',
        'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 'scholar.google',
        'researchgate.net', 'jstor.org',
    ],
    # 主流媒體
    SourceCredibility.MAJOR_NEWS: [
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'cnn.com', 'bloomberg.com', 'wsj.com', 'economist.com',
        # 台灣主流媒體
        'udn.com', 'ltn.com.tw', 'chinatimes.com', 'cna.com.tw',
        'tvbs.com.tw', 'ettoday.net', 'setn.com',
        # 中國主流媒體
        'xinhuanet.com', 'people.com.cn', 'caixin.com',
        # 日本
        'nhk.or.jp', 'asahi.com', 'nikkei.com',
    ],
    # 維基百科
    SourceCredibility.WIKIPEDIA: [
        'wikipedia.org', 'wikimedia.org', 'wikidata.org',
    ],
    # 科技媒體
    SourceCredibility.TECH_MEDIA: [
        'techcrunch.com', 'theverge.com', 'wired.com', 'arstechnica.com',
        'engadget.com', 'cnet.com', 'zdnet.com', 'venturebeat.com',
        # 台灣科技媒體
        'bnext.com.tw', 'technews.tw', 'ithome.com.tw', 'inside.com.tw',
    ],
    # 內容農場黑名單
    SourceCredibility.CONTENT_FARM: [
        'kknews.cc', 'read01.com', 'twgreatdaily.com',
        'bomb01.com', 'coco01.today', 'how01.com',
        'ptt01.cc', 'life.tw', 'push01.net',
    ],
}


def get_source_credibility(url: str) -> Tuple[SourceCredibility, int]:
    """
    評估來源可信度
    回傳：(可信度等級, 分數 0-5)
    """
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
    except:
        return SourceCredibility.UNKNOWN, 1
    
    # 檢查各類別
    for credibility, domains in CREDIBILITY_DOMAINS.items():
        for d in domains:
            if d in domain:
                return credibility, credibility.value
    
    # 特殊規則
    if '.gov' in domain or '.edu' in domain:
        return SourceCredibility.OFFICIAL, 5
    if 'wiki' in domain:
        return SourceCredibility.WIKIPEDIA, 3
    if 'news' in domain or 'times' in domain:
        return SourceCredibility.LOCAL_NEWS, 3
    
    return SourceCredibility.UNKNOWN, 1


# ============================================================
# 3. 資料結構定義
# ============================================================

@dataclass
class FactClaim:
    """事實聲明"""
    claim: str
    sources: List[str] = field(default_factory=list)
    contradicting_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verified: bool = False
    verification_notes: str = ""


@dataclass
class SourceInfo:
    """來源資訊"""
    url: str
    title: str
    content: str
    credibility: SourceCredibility
    credibility_score: int
    extracted_date: Optional[str] = None
    extraction_time: float = field(default_factory=time.time)
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['credibility'] = self.credibility.name
        return d


@dataclass
class CacheEntry:
    """快取項目結構"""
    question: str
    answer: str
    sources: List[str]
    timestamp: float
    confidence: float
    fact_claims: List[dict] = field(default_factory=list)
    
    def is_expired(self, ttl: int = CACHE_TTL) -> bool:
        return time.time() - self.timestamp > ttl
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "CacheEntry":
        return cls(**data)


# ============================================================
# 4. 持久化快取系統（優化方式）
# ============================================================

class PersistentCache:
    """持久化快取管理器"""
    
    def __init__(self, cache_dir: Path = CACHE_DIR, ttl: int = CACHE_TTL):
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[str, CacheEntry] = {}
    
    def _get_cache_key(self, question: str) -> str:
        normalized = " ".join(question.strip().split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"
    
    def get(self, question: str) -> Optional[CacheEntry]:
        key = self._get_cache_key(question)
        
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if not entry.is_expired(self.ttl):
                return entry
            else:
                del self._memory_cache[key]
        
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text(encoding='utf-8'))
                entry = CacheEntry.from_dict(data)
                if not entry.is_expired(self.ttl):
                    self._memory_cache[key] = entry
                    return entry
                else:
                    cache_path.unlink()
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"    快取讀取失敗: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, question: str, answer: str, sources: List[str], 
            confidence: float = 0.8, fact_claims: List[dict] = None):
        key = self._get_cache_key(question)
        entry = CacheEntry(
            question=question,
            answer=answer,
            sources=sources,
            timestamp=time.time(),
            confidence=confidence,
            fact_claims=fact_claims or []
        )
        
        self._memory_cache[key] = entry
        cache_path = self._get_cache_path(key)
        cache_path.write_text(
            json.dumps(entry.to_dict(), ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
    
    def clear_expired(self) -> int:
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
        total = len(list(self.cache_dir.glob("*.json")))
        memory_count = len(self._memory_cache)
        return {
            "total_cached": total,
            "in_memory": memory_count,
            "cache_dir": str(self.cache_dir),
            "ttl_hours": self.ttl / 3600
        }


# 初始化快取
cache = PersistentCache()


# ============================================================
# 5. 搜尋工具函數
# ============================================================

def search_searxng(query: str, limit: int = 5, retries: int = MAX_RETRIES) -> List[dict]:
    """
    執行 SearXNG 搜尋（含重試機制）
    """
    print(f"    🔍 搜尋: {query}")
    
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
                results = []
                seen_urls = set()
                
                for r in data.get('results', []):
                    url = r.get('url', '')
                    if url and url not in seen_urls:
                        # 跳過內容農場
                        credibility, score = get_source_credibility(url)
                        if credibility == SourceCredibility.CONTENT_FARM:
                            continue
                        
                        seen_urls.add(url)
                        results.append({
                            'title': r.get('title', '無標題'),
                            'url': url,
                            'content': r.get('content', ''),
                            'engine': r.get('engine', 'unknown'),
                            'credibility': credibility.name,
                            'credibility_score': score
                        })
                
                # 按可信度排序
                results.sort(key=lambda x: x['credibility_score'], reverse=True)
                
                print(f"    ✓ 找到 {len(results)} 筆結果（已過濾內容農場）")
                return results[:limit]
            
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                print(f"    ⏳ Rate limited，等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠ HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            print(f"    ⏳ 逾時，重試 {attempt + 1}/{retries}")
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            print(f"    ❌ 連線錯誤: {e}")
            time.sleep(1)
    
    return []


def generate_multi_angle_queries(question: str, existing_queries: List[str]) -> List[str]:
    """
    生成多角度搜尋關鍵字
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是搜尋策略專家。針對使用者的問題，生成 3 個不同角度的搜尋關鍵字。

問題：{question}

已經搜尋過（避免重複）：{existing}

生成策略：
1. 直接關鍵字：問題的核心主題
2. 驗證角度：尋找官方或權威來源
3. 反面驗證：搜尋可能的反駁或不同觀點

請用 JSON 陣列格式回傳 3 個關鍵字，例如：
["關鍵字1", "關鍵字2", "關鍵字3"]

只輸出 JSON，不要其他文字。""")
    ])
    
    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "existing": ", ".join(existing_queries) if existing_queries else "（無）"
        })
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        
        queries = json.loads(response.strip())
        new_queries = [q for q in queries if q not in existing_queries]
        return new_queries[:3]
        
    except Exception as e:
        print(f"    ⚠ 關鍵字生成失敗: {e}")
        return [question] if question not in existing_queries else []


def generate_contradiction_queries(
    question: str, 
    contradictions: List[str],
    existing_queries: List[str]
) -> List[str]:
    """
    針對矛盾生成驗證關鍵字
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你需要解決查證過程中發現的矛盾。

原始問題：{question}

發現的矛盾：
{contradictions}

已搜尋過：{existing}

請生成 2 個針對性的搜尋關鍵字，用於：
1. 找到更權威的來源來確認事實
2. 找到最新的資訊來解決矛盾

用 JSON 陣列格式回傳：["關鍵字1", "關鍵字2"]
只輸出 JSON。""")
    ])
    
    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "question": question,
            "contradictions": "\n".join(contradictions),
            "existing": ", ".join(existing_queries)
        })
        
        response = response.strip()
        if "```" in response:
            response = response.split("```")[1] if "```json" not in response else response.split("```json")[1]
            response = response.split("```")[0]
        
        queries = json.loads(response.strip())
        return [q for q in queries if q not in existing_queries][:2]
    except:
        return []


# ============================================================
# 6. VLM 視覺閱讀
# ============================================================

async def vlm_read_single_page(
    browser: Browser, 
    url: str, 
    title: str,
    max_screenshots: int = 3
) -> SourceInfo:
    """
    非同步讀取單一網頁
    """
    credibility, score = get_source_credibility(url)
    
    result = SourceInfo(
        url=url,
        title=title,
        content="",
        credibility=credibility,
        credibility_score=score,
        success=False
    )
    
    context = None
    try:
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 1200},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        )
        page = await context.new_page()
        
        # 設定請求攔截
        await page.route("**/*", lambda route: (
            route.abort() if any(x in route.request.url for x in [
                'analytics', 'tracking', 'ads', 'doubleclick', 
                'facebook.com/tr', 'googlesyndication', 'adservice'
            ]) else route.continue_()
        ))
        
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        await page.wait_for_timeout(2500)
        
        # 嘗試提取發布日期
        extracted_date = await extract_publish_date(page)
        result.extracted_date = extracted_date
        
        # 注入 CSS 隱藏干擾元素
        await page.add_style_tag(content="""
            iframe, .ad, .ads, .advertisement, 
            [class*="cookie"], [class*="popup"], 
            [class*="modal"], [class*="overlay"],
            [class*="banner"], [id*="banner"],
            header, footer, nav, aside,
            [class*="sidebar"], [class*="related"],
            [class*="recommend"], [class*="comment"] {
                opacity: 0 !important;
                pointer-events: none !important;
            }
        """)
        
        # 滾動截圖
        screenshots_b64 = []
        for i in range(max_screenshots):
            screenshot = await page.screenshot(type='png')
            b64 = base64.b64encode(screenshot).decode('utf-8')
            screenshots_b64.append(b64)
            await page.evaluate("window.scrollBy(0, 900)")
            await page.wait_for_timeout(600)
        
        # 使用 VLM 分析
        if screenshots_b64:
            content = await analyze_screenshots_with_vlm(screenshots_b64, title, credibility.name)
            result.content = content
            result.success = True
            
    except Exception as e:
        result.error = str(e)
        print(f"    ❌ 讀取失敗 ({title[:30]}...): {type(e).__name__}")
    finally:
        if context:
            await context.close()
    
    return result


async def extract_publish_date(page: Page) -> Optional[str]:
    """
    嘗試從網頁提取發布日期
    """
    try:
        selectors = [
            'time[datetime]',
            '[class*="date"]',
            '[class*="time"]',
            'meta[property="article:published_time"]',
            'meta[name="publishdate"]',
        ]
        
        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    datetime_attr = await element.get_attribute('datetime')
                    if datetime_attr:
                        return datetime_attr[:10]
                    
                    content_attr = await element.get_attribute('content')
                    if content_attr:
                        return content_attr[:10]
                    
                    text = await element.text_content()
                    if text and len(text) < 50:
                        return text.strip()
            except:
                continue
        
        return None
    except:
        return None


async def analyze_screenshots_with_vlm(
    screenshots_b64: List[str], 
    title: str,
    credibility: str
) -> str:
    """
    VLM 分析截圖內容
    """
    msg_content = [
        {
            "type": "text",
            "text": f"""這是「{title}」網頁的截圖（來源可信度：{credibility}）。

請仔細閱讀並提取以下資訊：

1. **核心事實**：文章的主要陳述和關鍵事實
2. **具體數據**：任何數字、日期、統計資料
3. **資訊來源**：文章引用的原始來源（如有）
4. **發布時間**：文章的發布或更新日期（如可見）
5. **作者/機構**：撰寫者或發布機構

注意事項：
- 只提取事實性內容，忽略廣告和意見
- 如果是新聞，區分事實報導和記者評論
- 如果有引用其他來源，請標註
- 使用繁體中文回答

請以結構化方式呈現提取的資訊。"""
        }
    ]
    
    for img in screenshots_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"}
        })
    
    try:
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
) -> List[SourceInfo]:
    """
    並行讀取多個網頁
    """
    if not urls:
        return []
    
    print(f"    📖 並行閱讀 {len(urls)} 個網頁...")
    
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
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            processed_results.append(SourceInfo(
                url=urls[i]['url'] if i < len(urls) else "",
                title=urls[i]['title'] if i < len(urls) else "",
                content=f"讀取失敗: {r}",
                credibility=SourceCredibility.UNKNOWN,
                credibility_score=0,
                success=False,
                error=str(r)
            ))
        else:
            processed_results.append(r)
    
    success_count = sum(1 for r in processed_results if r.success)
    print(f"    ✓ 完成，成功 {success_count}/{len(urls)}")
    
    return processed_results


# ============================================================
# 7. 事實提取與驗證
# ============================================================

def extract_fact_claims(question: str, sources: List[SourceInfo]) -> List[FactClaim]:
    """
    從來源中提取可驗證的事實聲明
    """
    if not sources:
        return []
    
    source_texts = []
    for i, s in enumerate(sources):
        if s.success and s.content:
            source_texts.append(f"""
【來源 {i+1}】{s.title}
可信度：{s.credibility.name} ({s.credibility_score}/5)
日期：{s.extracted_date or '未知'}
URL：{s.url}
內容摘要：
{s.content[:2000]}
""")
    
    if not source_texts:
        return []
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是事實查核專家。請從提供的來源中提取與問題相關的「事實聲明」。

問題：{question}

來源資訊：
{sources}

請提取 3-5 個關鍵事實聲明，用 JSON 格式回傳：
[
    {{
        "claim": "具體的事實聲明",
        "supporting_sources": [來源編號列表，如 [1, 3]],
        "contradicting_sources": [如有矛盾的來源編號],
        "confidence": 0.0-1.0 的信心度,
        "notes": "備註，如來源間的差異"
    }}
]

評估標準：
- 多個獨立來源支持 → 高信心度
- 僅單一來源 → 中等信心度
- 來源間有矛盾 → 低信心度，需標註

只輸出 JSON，不要其他文字。""")
    ])
    
    try:
        response = (prompt | llm_reasoning | StrOutputParser()).invoke({
            "question": question,
            "sources": "\n".join(source_texts)
        })
        
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        
        claims_data = json.loads(response.strip())
        
        claims = []
        for c in claims_data:
            supporting = c.get('supporting_sources', [])
            contradicting = c.get('contradicting_sources', [])
            
            claim = FactClaim(
                claim=c.get('claim', ''),
                sources=[sources[i-1].url for i in supporting if 0 < i <= len(sources)],
                contradicting_sources=[sources[i-1].url for i in contradicting if 0 < i <= len(sources)],
                confidence=c.get('confidence', 0.5),
                verified=len(supporting) >= 2,
                verification_notes=c.get('notes', '')
            )
            claims.append(claim)
        
        return claims
        
    except Exception as e:
        print(f"    ⚠ 事實提取失敗: {e}")
        return []


def detect_contradictions(claims: List[FactClaim]) -> Tuple[bool, List[str]]:
    """
    偵測事實聲明中的矛盾
    """
    contradictions = []
    
    for claim in claims:
        if claim.contradicting_sources:
            contradictions.append(
                f"「{claim.claim}」- 有 {len(claim.contradicting_sources)} 個來源提出不同說法"
            )
        if claim.confidence < 0.5 and claim.verification_notes:
            contradictions.append(
                f"「{claim.claim}」- {claim.verification_notes}"
            )
    
    return len(contradictions) > 0, contradictions


def calculate_confidence(
    sources: List[dict], 
    claims: List[dict], 
    contradictions: List[str],
    credibility_score: float
) -> float:
    """計算整體信心度"""
    if not sources:
        return 0.0
    
    # 基礎分數：來源可信度 (0-40%)
    base_score = min(credibility_score / 5 * 0.4, 0.4)
    
    # 來源數量分數 (0-20%)
    source_score = min(len(sources) / 5 * 0.2, 0.2)
    
    # 事實驗證分數 (0-30%)
    if claims:
        verified_ratio = sum(1 for c in claims if c.get('verified', False)) / len(claims)
        claim_score = verified_ratio * 0.3
    else:
        claim_score = 0.1
    
    # 矛盾懲罰 (0-10%)
    contradiction_penalty = min(len(contradictions) * 0.05, 0.1)
    
    confidence = base_score + source_score + claim_score - contradiction_penalty
    return max(0.1, min(confidence, 1.0))


# ============================================================
# 8. 定義 State
# ============================================================

class AgentState(TypedDict):
    # 基本欄位
    question: str
    knowledge_base: str
    current_queries: List[str]
    search_round: int
    final_answer: str
    is_sufficient: bool
    
    # 來源追蹤
    all_sources: List[dict]
    search_history: List[str]
    
    # 事實驗證
    fact_claims: List[dict]
    contradictions: List[str]
    
    # 評估結果
    confidence: float
    credibility_score: float
    verification_status: str


# ============================================================
# 9. 定義 Nodes（符合規定：planner, query_gen, search_tool）
# ============================================================

def check_cache_node(state: AgentState) -> dict:
    """[Node] 快取檢查（優化機制）"""
    question = state["question"]
    print(f"\n{'='*60}")
    print(f"📋 問題: {question}")
    print(f"{'='*60}")
    print(f"\n[Cache] 🗄️ 檢查快取...")
    
    cached = cache.get(question)
    
    if cached:
        print(f"    ✓ 命中快取！(信心度: {cached.confidence:.0%})")
        return {
            "final_answer": cached.answer,
            "all_sources": [{"url": s} for s in cached.sources],
            "confidence": cached.confidence,
            "fact_claims": cached.fact_claims,
            "is_sufficient": True,
            "verification_status": "cached"
        }
    else:
        print("    ✗ 無快取，開始查證流程")
        return {
            "knowledge_base": "",
            "search_round": 0,
            "all_sources": [],
            "search_history": [],
            "fact_claims": [],
            "contradictions": [],
            "confidence": 0.0,
            "credibility_score": 0.0,
            "is_sufficient": False,
            "verification_status": "pending"
        }


def planner_node(state: AgentState) -> dict:
    """
    [Node: planner] 決策節點
    
    規定要求：LLM 會判斷當前蒐集的資訊是否足以回應使用者
    """
    print(f"\n[Planner] 📊 評估查證進度 (輪次 {state.get('search_round', 0)}/{MAX_SEARCH_ROUNDS})...")
    
    search_round = state.get("search_round", 0)
    sources = state.get("all_sources", [])
    claims = state.get("fact_claims", [])
    contradictions = state.get("contradictions", [])
    credibility_score = state.get("credibility_score", 0)
    
    # 達到最大輪數
    if search_round >= MAX_SEARCH_ROUNDS:
        print(f"    達到最大搜尋輪數 ({MAX_SEARCH_ROUNDS})")
        return {"is_sufficient": True}
    
    # 第一輪，尚未搜尋
    if search_round == 0 and not sources:
        print("    首次查詢，需要搜尋資訊")
        return {"is_sufficient": False}
    
    # 沒有來源
    if not sources:
        print("    無來源，繼續搜尋...")
        return {"is_sufficient": False}
    
    # 使用 LLM 進行決策評估
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個嚴謹的查證決策者。請評估目前收集的資訊是否足以回答問題。

使用者問題：{question}

目前收集到的資訊摘要：
- 來源數量：{source_count}
- 平均可信度：{credibility}/5
- 已驗證事實數：{verified_claims}
- 高信心事實數：{high_confidence_claims}
- 發現的矛盾數：{contradiction_count}

資訊內容預覽：
{knowledge_preview}

請用 JSON 格式回應：
{{
    "completeness": <1-10 資訊完整度>,
    "credibility": <1-10 來源可信度>,
    "need_more_search": <true/false>,
    "reason": "<簡短說明理由>"
}}

評估標準：
- completeness >= 7 且 credibility >= 6 才算足夠
- 如有未解決的矛盾，應繼續搜尋
- 如果只有單一來源，可信度應降低

只輸出 JSON，不要其他文字。""")
    ])
    
    source_count = len(sources)
    verified_claims = sum(1 for c in claims if c.get('verified', False))
    high_confidence_claims = sum(1 for c in claims if c.get('confidence', 0) >= 0.7)
    
    try:
        response = (prompt | llm | StrOutputParser()).invoke({
            "question": state["question"],
            "source_count": source_count,
            "credibility": f"{credibility_score:.1f}",
            "verified_claims": verified_claims,
            "high_confidence_claims": high_confidence_claims,
            "contradiction_count": len(contradictions),
            "knowledge_preview": state.get("knowledge_base", "")[:1500]
        })
        
        # 解析 JSON
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
        if response.endswith("```"):
            response = response.rsplit("```", 1)[0]
        
        analysis = json.loads(response.strip())
        
        completeness = analysis.get("completeness", 0)
        cred = analysis.get("credibility", 0)
        need_more = analysis.get("need_more_search", True)
        reason = analysis.get("reason", "")
        
        print(f"    完整度: {completeness}/10, 可信度: {cred}/10")
        print(f"    理由: {reason}")
        
        # 判斷是否足夠
        is_sufficient = (completeness >= 7 and cred >= 6) and not need_more
        
        # 如果有矛盾且還有搜尋次數，繼續搜尋
        if contradictions and search_round < MAX_SEARCH_ROUNDS - 1:
            is_sufficient = False
            print("    → 需要解決矛盾，繼續搜尋")
        
        # 計算信心度
        confidence = calculate_confidence(sources, claims, contradictions, credibility_score)
        
        if is_sufficient:
            print(f"    ✓ 資訊足夠，準備生成報告 (信心度: {confidence:.0%})")
        else:
            print(f"    ✗ 資訊不足，繼續搜尋 (當前信心度: {confidence:.0%})")
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"    ⚠ 決策分析失敗: {e}，使用預設邏輯")
        
        # 預設邏輯
        is_sufficient = False
        if source_count >= MIN_SOURCES_FOR_CONFIDENCE and credibility_score >= 3:
            if not contradictions or search_round >= 2:
                is_sufficient = True
        
        confidence = calculate_confidence(sources, claims, contradictions, credibility_score)
        
        return {
            "is_sufficient": is_sufficient,
            "confidence": confidence
        }


def query_gen_node(state: AgentState) -> dict:
    """
    [Node: query_gen] 關鍵字生成節點
    
    規定要求：LLM 會分析使用者的問題生成檢索關鍵字
    """
    print(f"\n[Query Gen] 🎯 生成搜尋關鍵字...")
    
    # 檢查是否有矛盾需要解決
    if state.get("contradictions"):
        print("    發現矛盾，生成驗證關鍵字...")
        contradiction_queries = generate_contradiction_queries(
            state["question"],
            state["contradictions"],
            state.get("search_history", [])
        )
        if contradiction_queries:
            for q in contradiction_queries:
                print(f"      • {q}")
            return {"current_queries": contradiction_queries}
    
    # 生成多角度搜尋關鍵字
    queries = generate_multi_angle_queries(
        state["question"],
        state.get("search_history", [])
    )
    
    if not queries:
        if state["question"] not in state.get("search_history", []):
            queries = [state["question"]]
        else:
            queries = []
    
    print(f"    生成 {len(queries)} 個搜尋關鍵字：")
    for q in queries:
        print(f"      • {q}")
    
    return {"current_queries": queries}


def search_tool_node(state: AgentState) -> dict:
    """
    [Node: search_tool] 搜尋工具節點
    
    規定要求：檢索+文字處理
    """
    print(f"\n[Search Tool] 🔎 執行搜尋與 VLM 閱讀...")
    
    queries = state.get("current_queries", [])
    if not queries:
        print("    ⚠ 無搜尋關鍵字")
        return {
            "search_round": state["search_round"] + 1
        }
    
    all_search_results = []
    new_history = list(state.get("search_history", []))
    
    # 執行所有搜尋
    for query in queries:
        if query in new_history:
            continue
        results = search_searxng(query, limit=4)
        all_search_results.extend(results)
        new_history.append(query)
    
    if not all_search_results:
        print("    ⚠ 未找到任何結果")
        return {
            "search_history": new_history,
            "search_round": state["search_round"] + 1
        }
    
    # 去重並按可信度排序
    seen_urls = set(s.get('url', '') for s in state.get("all_sources", []))
    unique_results = []
    for r in all_search_results:
        if r['url'] not in seen_urls:
            seen_urls.add(r['url'])
            unique_results.append(r)
    
    unique_results.sort(key=lambda x: x.get('credibility_score', 0), reverse=True)
    
    # 選取要讀取的網頁
    urls_to_read = unique_results[:VLM_READ_COUNT]
    
    print(f"    選取 {len(urls_to_read)} 個網頁進行深度閱讀：")
    for u in urls_to_read:
        print(f"      • [{u.get('credibility', '?')}] {u['title'][:40]}...")
    
    # 並行 VLM 閱讀
    vlm_results = asyncio.run(vlm_read_websites_parallel(urls_to_read))
    
    # 更新知識庫
    new_kb = state.get("knowledge_base", "")
    new_sources = list(state.get("all_sources", []))
    
    for source in vlm_results:
        if source.success and source.content:
            new_kb += f"""

══════════════════════════════════════
📰 {source.title}
🔗 {source.url}
📊 可信度: {source.credibility.name} ({source.credibility_score}/5)
📅 日期: {source.extracted_date or '未知'}
══════════════════════════════════════
{source.content}
"""
            new_sources.append(source.to_dict())
    
    # 計算平均可信度分數
    if new_sources:
        avg_credibility = sum(s.get('credibility_score', 0) for s in new_sources) / len(new_sources)
    else:
        avg_credibility = 0
    
    return {
        "knowledge_base": new_kb,
        "all_sources": new_sources,
        "search_history": new_history,
        "search_round": state["search_round"] + 1,
        "credibility_score": avg_credibility
    }


def fact_extraction_node(state: AgentState) -> dict:
    """[Node] 事實提取與矛盾偵測"""
    print(f"\n[Fact Extraction] 🔬 提取事實聲明...")
    
    # 將 dict 轉換為 SourceInfo
    sources = []
    for s in state.get("all_sources", []):
        try:
            source = SourceInfo(
                url=s.get('url', ''),
                title=s.get('title', ''),
                content=s.get('content', ''),
                credibility=SourceCredibility[s.get('credibility', 'UNKNOWN')],
                credibility_score=s.get('credibility_score', 0),
                extracted_date=s.get('extracted_date'),
                success=s.get('success', True)
            )
            sources.append(source)
        except:
            continue
    
    if not sources:
        return {
            "fact_claims": [],
            "contradictions": []
        }
    
    # 提取事實聲明
    claims = extract_fact_claims(state["question"], sources)
    
    # 偵測矛盾
    has_contradictions, contradiction_list = detect_contradictions(claims)
    
    print(f"    提取了 {len(claims)} 個事實聲明")
    for c in claims:
        status = "✓" if c.verified else "?"
        print(f"      {status} {c.claim[:50]}... (信心度: {c.confidence:.0%})")
    
    if has_contradictions:
        print(f"    ⚠ 發現 {len(contradiction_list)} 個矛盾")
        for cont in contradiction_list:
            print(f"      • {cont[:60]}...")
    
    return {
        "fact_claims": [asdict(c) for c in claims],
        "contradictions": contradiction_list
    }


def final_answer_node(state: AgentState) -> dict:
    """[Node] 產生最終報告"""
    print(f"\n[Final Answer] 📝 生成查證報告...")
    
    claims = state.get("fact_claims", [])
    sources = state.get("all_sources", [])
    contradictions = state.get("contradictions", [])
    
    # 準備來源列表
    source_list = []
    for s in sources:
        cred = s.get('credibility', 'UNKNOWN')
        score = s.get('credibility_score', 0)
        date = s.get('extracted_date', '未知')
        source_list.append(f"• [{cred} {score}/5] {s.get('title', '未知')} ({date})\n  {s.get('url', '')}")
    
    # 準備事實聲明摘要
    claims_summary = []
    for c in claims:
        status = "✅ 已驗證" if c.get('verified') else "⚠️ 待確認"
        conf = c.get('confidence', 0)
        claims_summary.append(f"{status} ({conf:.0%}) {c.get('claim', '')}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是專業的事實查核報告撰寫者。請根據查證結果撰寫報告。

問題：{question}

已收集資訊：
{knowledge}

事實聲明驗證結果：
{claims}

發現的矛盾或不一致：
{contradictions}

參考來源：
{sources}

請撰寫一份完整的查證報告，包含：

## 查證結論
（一句話總結查證結果）

## 關鍵發現
（列出最重要的事實，標註信心度）

## 詳細分析
（解釋查證過程和推理邏輯）

## 來源評估
（評估來源的可信度和一致性）

## 注意事項
（如有矛盾或不確定之處，請說明）

## 參考來源
（列出主要來源）

請用繁體中文撰寫，保持客觀中立。""")
    ])
    
    answer = (prompt | llm_reasoning | StrOutputParser()).invoke({
        "question": state["question"],
        "knowledge": state.get("knowledge_base", "（無資訊）")[:8000],
        "claims": "\n".join(claims_summary) if claims_summary else "（未提取事實聲明）",
        "contradictions": "\n".join(contradictions) if contradictions else "（無矛盾）",
        "sources": "\n".join(source_list) if source_list else "（無來源）"
    })
    
    # 計算最終信心度
    confidence = state.get("confidence", 0.5)
    
    # 決定驗證狀態
    if confidence >= 0.7:
        verification_status = "verified"
    elif confidence >= 0.5:
        verification_status = "partially_verified"
    else:
        verification_status = "unverified"
    
    # 寫入快取
    source_urls = [s.get('url', '') for s in sources if s.get('url')]
    cache.set(
        question=state["question"],
        answer=answer,
        sources=source_urls,
        confidence=confidence,
        fact_claims=claims
    )
    
    print(f"    ✓ 報告已生成並存入快取")
    
    return {
        "final_answer": answer,
        "confidence": confidence,
        "verification_status": verification_status
    }


# ============================================================
# 10. 建立 Graph（符合規定的節點名稱）
# ============================================================

workflow = StateGraph(AgentState)

# 加入節點（符合規定：planner, query_gen, search_tool）
workflow.add_node("check_cache", check_cache_node)      # 優化：快取機制
workflow.add_node("planner", planner_node)              # 規定節點：決策
workflow.add_node("query_gen", query_gen_node)          # 規定節點：關鍵字生成
workflow.add_node("search_tool", search_tool_node)      # 規定節點：搜尋工具
workflow.add_node("fact_extraction", fact_extraction_node)  # 額外：事實提取
workflow.add_node("final_answer", final_answer_node)    # 最終答案

# 設定進入點
workflow.set_entry_point("check_cache")


# 條件邊函式
def route_cache(state: AgentState) -> str:
    if state.get("final_answer"):
        return "end"
    return "planner"


def route_planner(state: AgentState) -> str:
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
workflow.add_edge("search_tool", "fact_extraction")
workflow.add_edge("fact_extraction", "planner")
workflow.add_edge("final_answer", END)

# 編譯
app = workflow.compile()


# ============================================================
# 11. 輔助函式與主程式
# ============================================================

def print_banner():
    print("""
╔═══════════════════════════════════════════════════════════════╗
║           🔍 自動查證 AI v1.1 (精準查證版)                      ║
╠═══════════════════════════════════════════════════════════════╣
║  符合課後練習規定：                                            ║
║  ✓ 優化方式：快取機制（check_cache 節點）                       ║
║  ✓ 必要節點：planner, query_gen, search_tool                   ║
╠═══════════════════════════════════════════════════════════════╣
║  全新功能：                                                    ║
║  ✓ 來源可信度評分（官方/學術/新聞/論壇...）                     ║
║  ✓ 交叉驗證機制（多來源確認事實）                               ║
║  ✓ 時效性檢查（提取發布日期）                                   ║
║  ✓ 矛盾偵測與解決（自動深入搜尋）                               ║
║  ✓ 事實聲明提取（結構化驗證）                                   ║
║  ✓ 多角度搜尋策略（直接/驗證/反面）                             ║
║  ✓ 內容農場過濾（自動排除低品質來源）                           ║
╠═══════════════════════════════════════════════════════════════╣
║  指令：                                                        ║
║  q / exit    - 離開程式                                        ║
║  /cache      - 查看快取統計                                    ║
║  /clear      - 清理過期快取                                    ║
║  /graph      - 顯示流程圖                                      ║
║  /domains    - 顯示可信來源網域                                 ║
╚═══════════════════════════════════════════════════════════════╝
""")


def handle_command(cmd: str) -> bool:
    cmd = cmd.strip().lower()
    
    if cmd == "/cache":
        stats = cache.get_stats()
        print(f"\n📊 快取統計：")
        print(f"   總快取數：{stats['total_cached']}")
        print(f"   記憶體中：{stats['in_memory']}")
        print(f"   快取目錄：{stats['cache_dir']}")
        print(f"   有效期：{stats['ttl_hours']:.1f} 小時")
        return True
    
    elif cmd == "/clear":
        cleared = cache.clear_expired()
        print(f"\n🗑️  已清理 {cleared} 筆過期快取")
        return True
    
    elif cmd == "/graph":
        print("\n📊 流程圖：")
        print(app.get_graph().draw_ascii())
        return True
    
    elif cmd == "/domains":
        print("\n📋 可信來源網域對照表：")
        for cred, domains in CREDIBILITY_DOMAINS.items():
            if cred != SourceCredibility.CONTENT_FARM:
                print(f"\n  【{cred.name}】(分數: {cred.value}/5)")
                for d in domains[:5]:
                    print(f"    • {d}")
                if len(domains) > 5:
                    print(f"    ... 等共 {len(domains)} 個")
        
        print(f"\n  【內容農場黑名單】(自動過濾)")
        for d in CREDIBILITY_DOMAINS[SourceCredibility.CONTENT_FARM][:5]:
            print(f"    • {d}")
        return True
    
    return False


def format_verification_status(status: str) -> str:
    """格式化驗證狀態"""
    status_map = {
        "verified": "✅ 已驗證",
        "partially_verified": "⚠️ 部分驗證",
        "unverified": "❓ 未驗證",
        "cached": "📦 來自快取"
    }
    return status_map.get(status, status)


if __name__ == "__main__":
    print_banner()
    
    # 啟動時清理過期快取
    cleared = cache.clear_expired()
    if cleared:
        print(f"🗑️  啟動時清理了 {cleared} 筆過期快取\n")
    
    print("-" * 65)
    
    while True:
        try:
            q = input("\n請輸入要查證的問題: ").strip()
            
            if not q:
                continue
            
            if q.lower() in ["q", "exit", "quit"]:
                print("\n👋 再見！")
                break
            
            if q.startswith("/"):
                if handle_command(q):
                    continue
            
            # 執行查證
            start_time = time.time()
            result = app.invoke({"question": q})
            elapsed = time.time() - start_time
            
            # 顯示結果
            print("\n" + "═" * 65)
            print("📋 查證報告")
            print("═" * 65)
            print(result["final_answer"])
            print("═" * 65)
            
            # 顯示統計
            print(f"\n📊 查證統計")
            print(f"   ⏱️  耗時: {elapsed:.1f} 秒")
            print(f"   🎯 信心度: {result.get('confidence', 0):.0%}")
            print(f"   📌 狀態: {format_verification_status(result.get('verification_status', 'unknown'))}")
            
            sources = result.get("all_sources", [])
            if sources:
                print(f"   📚 來源數: {len(sources)} 個")
                avg_cred = sum(s.get('credibility_score', 0) for s in sources) / len(sources)
                print(f"   ⭐ 平均可信度: {avg_cred:.1f}/5")
            
            claims = result.get("fact_claims", [])
            if claims:
                verified = sum(1 for c in claims if c.get('verified'))
                print(f"   ✓ 已驗證事實: {verified}/{len(claims)}")
            
            print("═" * 65)
            
        except KeyboardInterrupt:
            print("\n\n👋 收到中斷信號，再見！")
            break
        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            import traceback
            traceback.print_exc()