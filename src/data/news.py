import yfinance as yf
from datetime import datetime
from typing import List, Dict
import feedparser
from urllib.parse import quote

_vader = None

def _get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader

_POSITIVE_PHRASES = [
    "better than expected", "above expectations", "above estimates",
    "record quarter", "record high", "record revenue", "record profit",
    "beat estimates", "beat expectations", "earnings beat",
    "raised guidance", "upgraded to", "outperform rating",
    "strong buy", "strong earnings", "strong quarter",
    "positive outlook", "favorable outlook", "growth story",
    "market leader", "industry leader", "competitive advantage",
    "accelerating growth", "improving margins", "expanding margins",
    "share buyback", "stock buyback", "dividend increase",
    "exceeded expectations", "well above", "well received",
    "game changer", "game-changing", "breakthrough",
    "all-time high", "historic high", "new high",
    "cost savings", "revenue growth", "profit growth",
    "expansion plans", "strategic acquisition", "strategic partnership",
    "joint venture", "merger", "acquisition", "acquired",
    "partnership", "partnership with", "teams up with", "partner with",
    "expansion into", "new market", "market expansion",
    "landmark deal", "major deal", "strategic deal",
    "acquires", "acquired", "merger",
    "bull market", "bull run",
    "analyst upgrade", "analyst upgrade",
    "price target raised", "price target increase",
    "strong demand", "high demand", "growing demand",
    "record sales", "strong sales", "sales growth",
]

_NEGATIVE_PHRASES = [
    "below expectations", "below estimates", "missed estimates",
    "missed expectations", "earnings miss", "revenue miss",
    "lower than expected", "worse than expected",
    "downgraded to", "underperform rating",
    "profit warning", "earnings warning", "revenue warning",
    "weak guidance", "lowered guidance", "cut guidance",
    "loss quarter", "declining revenue", "declining sales",
    "workforce reduction", "massive layoff",
    "accounting issues", "regulatory issues", "legal issues",
    "class action", "shareholder lawsuit",
    "debt downgrade", "credit downgrade", "rating downgrade",
    "negative outlook", "negative trend",
    "market share loss", "losing market share",
    "supply chain issues", "supply chain disruption",
    "demand weakness", "weakening demand", "slowing demand",
    "recession fears", "economic slowdown", "economic contraction",
    "trade war", "tariff", "sanctions",
    "investigation", "regulatory scrutiny", "antitrust",
    "data breach", "cyber attack", "cybersecurity",
    "missed deadline", "delay", "delayed", "halt", "halts",
    "production halt", "recall", "product recall",
    "strike", "walkout", "union dispute",
    "government shutdown", "debt ceiling",
    "price target cut", "price target lowered",
    "analyst downgrade", "analyst downgrade",
    "weak demand", "low demand", "falling demand",
    "bear market", "bear run",
    "sales decline", "revenue decline",
    "warns of", "warning of",
    "court ruling", "adverse ruling",
    "investor concern", "shareholder concern",
]


def _score_headline(title: str) -> float:
    vader = _get_vader()
    scores = vader.polarity_scores(title)
    base = scores["compound"]
    title_lower = title.lower()
    phrase_boost = 0.0
    for phrase in _POSITIVE_PHRASES:
        if phrase in title_lower:
            phrase_boost += 0.25
    for phrase in _NEGATIVE_PHRASES:
        if phrase in title_lower:
            phrase_boost -= 0.25
    result = base + phrase_boost * 0.6
    return max(-1.0, min(1.0, result))


def _extract(item: dict) -> dict:
    content = item.get("content") or item
    if not isinstance(content, dict):
        content = item
    title = content.get("title", "")
    if not title:
        title = content.get("headline", "")
    pub = content.get("pubDate") or content.get("publication_date") or ""
    if isinstance(pub, str) and pub:
        try:
            pub = datetime.strptime(pub.replace("Z", ""), "%Y-%m-%dT%H:%M:%S").strftime("%m/%d %H:%M")
        except ValueError:
            pub = pub[:16]
    provider = content.get("provider", {})
    if isinstance(provider, dict):
        publisher = provider.get("displayName", "") or provider.get("name", "")
    elif isinstance(provider, str):
        publisher = provider
    else:
        publisher = ""
    c_url = content.get("canonicalUrl", {})
    link = c_url.get("url", "") if isinstance(c_url, dict) else ""
    return {"title": title, "publisher": publisher, "date": pub, "link": link,
            "sentiment": _score_headline(title)}


def _extract_rss(entry) -> dict:
    title = entry.get("title", "")
    pub = entry.get("published", "")[:16]
    source = entry.get("source", {})
    if hasattr(source, "title"):
        publisher = source.title
    else:
        publisher = source.get("title", "") if isinstance(source, dict) else ""
    link = entry.get("link", "")
    return {"title": title, "publisher": publisher, "date": pub, "link": link,
            "sentiment": _score_headline(title)}


def fetch_news(ticker: str, max_articles: int = 10) -> List[Dict]:
    articles = []
    try:
        tk = yf.Ticker(ticker)
        raw = tk.news or []
        articles = [_extract(item) for item in raw if isinstance(item, dict) and
                    (item.get("content") or item).get("title")]
    except Exception:
        pass
    try:
        rss_url = f"https://news.google.com/rss/search?q={quote(ticker + ' stock')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        seen = {a["title"].lower() for a in articles}
        for entry in feed.entries:
            if len(articles) >= max_articles * 2:
                break
            title_lower = entry.title.lower()
            if title_lower not in seen:
                seen.add(title_lower)
                articles.append(_extract_rss(entry))
    except Exception:
        pass
    return articles[:max_articles]


def aggregate_sentiment(articles: List[Dict]) -> Dict:
    if not articles:
        return {"avg": 0.0, "pos": 0, "neg": 0, "neutral": 0, "total": 0}
    scores = [a["sentiment"] for a in articles]
    return {
        "avg": round(sum(scores) / len(scores), 3),
        "pos": sum(1 for s in scores if s > 0.15),
        "neg": sum(1 for s in scores if s < -0.15),
        "neutral": sum(1 for s in scores if -0.15 <= s <= 0.15),
        "total": len(articles),
    }


def sentiment_label(score: float) -> str:
    if score > 0.15:
        return "BULLISH"
    if score < -0.15:
        return "BEARISH"
    return "NEUTRAL"
