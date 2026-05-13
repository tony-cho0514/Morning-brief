"""Fetch US stock market snapshots and financial news headlines."""


import logging
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Iterable

import feedparser
import yfinance as yf

import ai_summarizer

log = logging.getLogger("newsletter.market")

DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "TSLA"]

# Tried in order; the first feed that returns ≥1 entry is used.
NEWS_FEEDS = [
    (
        "https://news.google.com/rss/search?"
        "q=US+stock+market+OR+SPY+OR+QQQ+OR+AAPL+OR+NVDA+OR+TSLA"
        "&hl=en-US&gl=US&ceid=US:en"
    ),
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.reuters.com/reuters/businessNews",
]
DEFAULT_NEWS_FEED = NEWS_FEEDS[0]

ACCENT = "#FF6B35"

# Famous investor quotes shown daily. Rotates by day-of-year.
QUOTES = [
    {
        "author_ko": "워런 버핏",
        "author_en": "Warren Buffett",
        "quote_ko": "남들이 탐욕스러울 때 두려워하고, 남들이 두려워할 때 탐욕스러워라.",
        "quote_en": "Be fearful when others are greedy, and greedy when others are fearful.",
    },
    {
        "author_ko": "피터 린치",
        "author_en": "Peter Lynch",
        "quote_ko": "당신이 무엇을 가지고 있는지, 왜 가지고 있는지를 알아라.",
        "quote_en": "Know what you own, and know why you own it.",
    },
    {
        "author_ko": "벤저민 그레이엄",
        "author_en": "Benjamin Graham",
        "quote_ko": "투자자의 가장 큰 적은 시장이 아니라 자기 자신이다.",
        "quote_en": "The investor's chief problem—and even his worst enemy—is likely to be himself.",
    },
    {
        "author_ko": "찰리 멍거",
        "author_en": "Charlie Munger",
        "quote_ko": "큰돈은 사고파는 데 있는 것이 아니라 기다리는 데 있다.",
        "quote_en": "The big money is not in the buying and selling, but in the waiting.",
    },
    {
        "author_ko": "존 보글",
        "author_en": "John Bogle",
        "quote_ko": "건초더미에서 바늘을 찾지 말라. 그냥 건초더미를 통째로 사라.",
        "quote_en": "Don't look for the needle in the haystack. Just buy the haystack.",
    },
    {
        "author_ko": "하워드 막스",
        "author_en": "Howard Marks",
        "quote_ko": "미래를 예측할 수는 없지만, 대비할 수는 있다.",
        "quote_en": "You can't predict. You can prepare.",
    },
    {
        "author_ko": "레이 달리오",
        "author_en": "Ray Dalio",
        "quote_ko": "수정구슬에 의지해 사는 사람은 깨진 유리를 먹게 된다.",
        "quote_en": "He who lives by the crystal ball will eat shattered glass.",
    },
    {
        "author_ko": "필립 피셔",
        "author_en": "Philip Fisher",
        "quote_ko": "주식 시장은 모든 것의 가격은 알아도 가치를 모르는 사람들로 가득하다.",
        "quote_en": "The stock market is filled with individuals who know the price of everything, but the value of nothing.",
    },
    {
        "author_ko": "제시 리버모어",
        "author_en": "Jesse Livermore",
        "quote_ko": "시장은 결코 틀리지 않는다. 틀리는 것은 사람의 의견일 뿐이다.",
        "quote_en": "Markets are never wrong, opinions often are.",
    },
    {
        "author_ko": "세스 클라먼",
        "author_en": "Seth Klarman",
        "quote_ko": "가치 투자는 본질적으로 역발상과 계산기의 결합이다.",
        "quote_en": "Value investing is at its core the marriage of a contrarian streak and a calculator.",
    },
]


def pick_quote(now: datetime) -> dict:
    return QUOTES[now.timetuple().tm_yday % len(QUOTES)]


@dataclass
class Quote:
    symbol: str
    name: str
    price: float | None
    change: float | None
    change_pct: float | None
    change_pct_1m: float | None = None
    reason: str = ""
    article_url: str = ""
    article_title: str = ""

    @property
    def is_up(self) -> bool:
        return (self.change or 0) >= 0

    @property
    def is_up_1m(self) -> bool:
        return (self.change_pct_1m or 0) >= 0


@dataclass
class Headline:
    title: str
    link: str
    source: str
    published: str
    title_ko: str = ""


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        if f != f:
            return None
        return f
    except (TypeError, ValueError):
        return None


def fetch_quotes(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    *,
    with_articles: bool = True,
    name_overrides: dict | None = None,
) -> list[Quote]:
    """Fetch ~1 month of history per ticker and compute daily + 1-month change.

    name_overrides: optional {symbol: display_name} mapping (used for crypto).
    with_articles: when False, skip yfinance ticker.news lookup (used for crypto).
    """
    quotes: list[Quote] = []
    for symbol in tickers:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo", auto_adjust=False)
            if hist.empty:
                log.warning("No history for %s", symbol)
                continue

            closes = hist["Close"]
            last_close = _safe_float(closes.iloc[-1])
            prev_close = (
                _safe_float(closes.iloc[-2])
                if len(closes) >= 2
                else _safe_float(hist["Open"].iloc[-1])
            )
            month_ago_close = (
                _safe_float(closes.iloc[0])
                if len(closes) >= 2
                else None
            )

            change = None
            change_pct = None
            if last_close is not None and prev_close is not None and prev_close != 0:
                change = last_close - prev_close
                change_pct = (change / prev_close) * 100

            change_pct_1m = None
            if (
                last_close is not None
                and month_ago_close is not None
                and month_ago_close != 0
            ):
                change_pct_1m = (last_close - month_ago_close) / month_ago_close * 100

            name = symbol
            if name_overrides and symbol in name_overrides:
                name = name_overrides[symbol]
            else:
                try:
                    info = ticker.fast_info
                    name = getattr(info, "shortName", None) or symbol
                except Exception:
                    pass

            article_title = ""
            article_url = ""
            if with_articles:
                article_title, article_url = _pick_top_article(ticker, symbol)

            quotes.append(
                Quote(
                    symbol=symbol,
                    name=name,
                    price=last_close,
                    change=change,
                    change_pct=change_pct,
                    change_pct_1m=change_pct_1m,
                    article_title=article_title,
                    article_url=article_url,
                )
            )
        except Exception as exc:
            log.warning("Failed to fetch %s: %s", symbol, exc)
    return quotes




def _pick_top_article(ticker, symbol: str) -> tuple[str, str]:
    """Return (title, url) of the freshest article for a ticker, or ("","")."""
    try:
        items = ticker.news or []
    except Exception as exc:
        log.warning("ticker.news failed for %s: %s", symbol, exc)
        return "", ""
    for item in items:
        # Newer yfinance: {"id": ..., "content": {"title": ..., "clickThroughUrl": {"url": ...}}}
        content = item.get("content") if isinstance(item, dict) else None
        if isinstance(content, dict):
            title = (content.get("title") or "").strip()
            url = ""
            for key in ("clickThroughUrl", "canonicalUrl"):
                ref = content.get(key)
                if isinstance(ref, dict) and ref.get("url"):
                    url = ref["url"]
                    break
            if title and url:
                return title, url
        # Legacy yfinance: {"title": ..., "link": ...}
        if isinstance(item, dict):
            title = (item.get("title") or "").strip()
            url = item.get("link") or ""
            if title and url:
                return title, url
    return "", ""


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _try_parse(url: str):
    try:
        return feedparser.parse(
            url,
            agent=USER_AGENT,
            request_headers={
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
    except Exception as exc:
        log.warning("Feed parse error for %s: %s", url, exc)
        return None


def fetch_headlines(
    feed_url: str | None = None,
    limit: int = 3,
    fallback_symbols: Iterable[str] | None = None,
) -> list[Headline]:
    """Try multiple RSS sources in order; return the first one with entries.

    `feed_url`, when provided, is tried first (preserves the existing
    `--news-feed` CLI override). Otherwise NEWS_FEEDS is used in order:
    Google News → BBC Business → Reuters Business.
    """
    candidates: list[str] = []
    if feed_url:
        candidates.append(feed_url)
        if fallback_symbols and "?" not in feed_url and "google.com" not in feed_url:
            symbols = ",".join(fallback_symbols)
            candidates.append(f"{feed_url}?s={symbols}&region=US&lang=en-US")
    for url in NEWS_FEEDS:
        if url not in candidates:
            candidates.append(url)

    parsed = None
    used_url = None
    for url in candidates:
        result = _try_parse(url)
        entries = getattr(result, "entries", None) if result else None
        status = getattr(result, "status", None) if result else None
        log.info(
            "Feed %s returned status=%s entries=%d",
            url, status, len(entries) if entries else 0,
        )
        if entries:
            parsed = result
            used_url = url
            break

    if not parsed or not parsed.entries:
        return []

    feed_title_default = parsed.feed.get("title") if parsed.feed else None
    if not feed_title_default:
        if used_url and "google.com" in used_url:
            feed_title_default = "Google News"
        elif used_url and "bbci" in used_url:
            feed_title_default = "BBC Business"
        elif used_url and "reuters" in used_url:
            feed_title_default = "Reuters Business"
        else:
            feed_title_default = "News"

    headlines: list[Headline] = []
    for entry in parsed.entries[:limit]:
        published = ""
        if getattr(entry, "published_parsed", None):
            try:
                published = datetime(*entry.published_parsed[:6]).strftime(
                    "%b %d, %Y %I:%M %p UTC"
                )
            except Exception:
                published = getattr(entry, "published", "")
        else:
            published = getattr(entry, "published", "")

        source = ""
        if getattr(entry, "source", None):
            source = entry.source.get("title", "") if isinstance(entry.source, dict) else ""
        if not source:
            source = feed_title_default

        headlines.append(
            Headline(
                title=getattr(entry, "title", "Untitled"),
                link=getattr(entry, "link", "#"),
                source=source,
                published=published,
            )
        )
    return headlines


def render_quotes_html(quotes: list[Quote]) -> str:
    """Card-per-stock layout. Header row has ticker + price/change pills,
    full-width analyst paragraph below, then article link.
    """
    if not quotes:
        return (
            '<p style="margin:0;color:#6b7280;font-size:14px;">'
            "현재 시장 데이터를 불러올 수 없습니다.</p>"
        )

    cards = []
    for q in quotes:
        up = q.is_up
        color = "#15803d" if up else "#b91c1c"
        bg = "#dcfce7" if up else "#fee2e2"
        arrow = "▲" if up else "▼"

        price_str = f"${q.price:,.2f}" if q.price is not None else "—"
        change_str = (
            f"{arrow} {q.change_pct:+.2f}%"
            if q.change_pct is not None
            else "—"
        )

        if q.change_pct_1m is not None:
            arrow_1m = "▲" if q.is_up_1m else "▼"
            color_1m = "#15803d" if q.is_up_1m else "#b91c1c"
            month_html = (
                f'<div style="margin-top:4px;font-size:10px;color:{color_1m};'
                f'font-weight:700;font-variant-numeric:tabular-nums;">'
                f'{arrow_1m} {q.change_pct_1m:+.2f}% <span style="color:#94a3b8;font-weight:500;">(1개월)</span></div>'
            )
        else:
            month_html = (
                '<div style="margin-top:4px;font-size:10px;color:#cbd5e1;">— (1개월)</div>'
            )

        reason_html = (
            f'<p style="margin:12px 0 0 0;font-size:13px;line-height:1.7;color:#334155;">'
            f'{escape(q.reason)}</p>'
            if q.reason
            else ""
        )

        article_html = ""
        if q.article_title and q.article_url:
            short = q.article_title if len(q.article_title) <= 90 else q.article_title[:87] + "…"
            article_html = (
                f'<div style="margin-top:10px;padding-top:10px;border-top:1px dashed #e2e8f0;'
                f'font-size:12px;line-height:1.4;">'
                f'<a href="{escape(q.article_url)}" '
                f'style="color:#2563eb;text-decoration:none;">📰 {escape(short)} →</a>'
                f'</div>'
            )

        cards.append(
            f"""
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" style="border-collapse:separate;background:#ffffff;border:1px solid #f1f5f9;border-radius:12px;margin-bottom:10px;">
              <tr>
                <td style="padding:14px 16px 12px 16px;">
                  <table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">
                    <tr>
                      <td style="vertical-align:top;">
                        <div style="font-weight:800;font-size:15px;color:#0f172a;letter-spacing:-0.01em;">{escape(q.symbol)}</div>
                        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">{escape(q.name)}</div>
                      </td>
                      <td style="text-align:right;vertical-align:top;white-space:nowrap;">
                        <div style="font-weight:800;color:#0f172a;font-size:15px;font-variant-numeric:tabular-nums;">{price_str}</div>
                        <div style="margin-top:5px;">
                          <span style="display:inline-block;padding:3px 9px;border-radius:999px;background:{bg};color:{color};font-size:11px;font-weight:700;font-variant-numeric:tabular-nums;">{change_str} <span style="font-weight:500;opacity:0.85;">오늘</span></span>
                        </div>
                        {month_html}
                      </td>
                    </tr>
                  </table>
                  {reason_html}
                  {article_html}
                </td>
              </tr>
            </table>
            """
        )

    return "".join(cards)


def render_summary_html(points: list[str]) -> str:
    if not points:
        return (
            '<p style="margin:0;color:#94a3b8;font-size:14px;">'
            "오늘의 핵심 요약을 준비하지 못했어요.</p>"
        )
    items = []
    for i, p in enumerate(points, 1):
        items.append(
            f"""
            <tr>
              <td valign="top" style="padding:6px 0;width:30px;">
                <div style="display:inline-block;width:22px;height:22px;line-height:22px;text-align:center;border-radius:999px;background:{ACCENT};color:#ffffff;font-size:11px;font-weight:700;">
                  {i}
                </div>
              </td>
              <td valign="top" style="padding:8px 0 8px 10px;font-size:15px;line-height:1.5;color:#0f172a;font-weight:600;">
                {escape(p)}
              </td>
            </tr>
            """
        )
    return (
        '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0">'
        f'<tbody>{"".join(items)}</tbody></table>'
    )


def render_summary_text(points: list[str]) -> str:
    if not points:
        return "오늘의 핵심 요약을 준비하지 못했어요."
    return "\n".join(f"  {i}. {p}" for i, p in enumerate(points, 1))


def render_lesson_html(lesson: str) -> str:
    """The 'oneulei sijang insight' card body — a daily AI-generated lesson
    grounded in today's actual market situation.
    """
    body = (
        lesson
        if lesson
        else "오늘의 인사이트를 준비하지 못했어요. 내일 더 좋은 글로 찾아올게요!"
    )
    return (
        '<div style="display:inline-block;padding:4px 10px;background:#fff1ea;color:#FF6B35;'
        'border-radius:6px;font-size:11px;font-weight:700;margin-bottom:12px;">'
        "💡 오늘의 인사이트</div>"
        f'<p style="margin:0;font-size:14px;line-height:1.8;color:#334155;">{escape(body)}</p>'
    )


def render_lesson_text(lesson: str) -> str:
    return (
        f"[오늘의 인사이트]\n"
        f"  {lesson if lesson else '오늘의 인사이트를 준비하지 못했어요.'}"
    )


def _unused_render_crypto_html(crypto: list[Quote]) -> str:
    """Compact crypto price table with 1-month change."""
    if not crypto:
        return (
            '<p style="margin:0;color:#94a3b8;font-size:14px;">'
            "코인 시세를 불러오지 못했어요.</p>"
        )
    rows = []
    for c in crypto:
        up = c.is_up
        color = "#15803d" if up else "#b91c1c"
        bg = "#dcfce7" if up else "#fee2e2"
        arrow = "▲" if up else "▼"
        price_str = f"${c.price:,.2f}" if c.price is not None else "—"
        change_str = (
            f"{arrow} {c.change_pct:+.2f}%"
            if c.change_pct is not None
            else "—"
        )
        if c.change_pct_1m is not None:
            color_1m = "#15803d" if c.is_up_1m else "#b91c1c"
            arrow_1m = "▲" if c.is_up_1m else "▼"
            month_str = f"{arrow_1m} {c.change_pct_1m:+.2f}%"
        else:
            color_1m = "#cbd5e1"
            month_str = "—"
        rows.append(
            f"""
            <tr>
              <td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;vertical-align:middle;">
                <div style="font-weight:700;font-size:14px;color:#0f172a;">{escape(c.symbol)}</div>
                <div style="font-size:11px;color:#94a3b8;">{escape(c.name)}</div>
              </td>
              <td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;text-align:right;font-weight:700;color:#0f172a;font-size:13px;font-variant-numeric:tabular-nums;white-space:nowrap;">
                {price_str}
              </td>
              <td style="padding:9px 12px;border-bottom:1px solid #f1f5f9;text-align:right;white-space:nowrap;">
                <div style="font-size:11px;font-weight:700;color:{color};font-variant-numeric:tabular-nums;">
                  <span style="display:inline-block;padding:2px 7px;border-radius:999px;background:{bg};">{change_str}</span>
                  <span style="color:#94a3b8;font-weight:500;margin-left:2px;">오늘</span>
                </div>
                <div style="margin-top:3px;font-size:10px;font-weight:700;color:{color_1m};font-variant-numeric:tabular-nums;">
                  {month_str} <span style="color:#cbd5e1;font-weight:500;">1개월</span>
                </div>
              </td>
            </tr>
            """
        )
    return (
        '<div style="display:inline-block;padding:4px 10px;background:#fff1ea;color:#FF6B35;'
        'border-radius:6px;font-size:11px;font-weight:700;margin-bottom:10px;">'
        "₿ 비트코인 & 주요 코인 시세</div>"
        '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" border="0" '
        'style="border-collapse:collapse;background:#ffffff;border:1px solid #f1f5f9;'
        'border-radius:10px;overflow:hidden;">'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def render_crypto_text(crypto: list[Quote]) -> str:
    if not crypto:
        return "코인 시세를 불러오지 못했어요."
    lines = ["[비트코인 & 주요 코인 시세]"]
    for c in crypto:
        price = f"${c.price:,.2f}" if c.price is not None else "—"
        day = (
            f"{'+' if c.is_up else ''}{c.change_pct:.2f}%"
            if c.change_pct is not None
            else "—"
        )
        mon = (
            f"{'+' if c.is_up_1m else ''}{c.change_pct_1m:.2f}%"
            if c.change_pct_1m is not None
            else "—"
        )
        lines.append(f"  {c.symbol:<5} {c.name:<6} {price:>12}  {day} (오늘) / {mon} (1개월)")
    return "\n".join(lines)


def render_concept_html(concept_term: str, explanation: str) -> str:
    """Concept explanation card body (Mon/Wed/Fri/weekends)."""
    body = (
        explanation
        if explanation
        else "오늘의 설명을 준비하지 못했어요. 내일 다시 찾아올게요!"
    )
    return (
        '<div style="display:inline-block;padding:4px 10px;background:#fff1ea;color:#FF6B35;'
        'border-radius:6px;font-size:11px;font-weight:700;margin-bottom:10px;">'
        f"오늘의 키워드 · {escape(concept_term)}</div>"
        f'<p style="margin:0;font-size:14px;line-height:1.75;color:#334155;">{escape(body)}</p>'
    )


def render_concept_text(concept_term: str, explanation: str) -> str:
    return (
        f"[오늘의 키워드: {concept_term}]\n"
        f"  {explanation if explanation else '오늘의 설명을 준비하지 못했어요.'}"
    )


def render_quote_block_html(quote_data: dict, context: str) -> str:
    quote_ko = quote_data.get("quote_ko", "")
    quote_en = quote_data.get("quote_en", "")
    author_ko = quote_data.get("author_ko", "")
    author_en = quote_data.get("author_en", "")
    context_html = (
        f'<div style="margin-top:12px;padding-top:12px;border-top:1px dashed #ffd2bb;'
        f'font-size:13px;line-height:1.6;color:#475569;">{escape(context)}</div>'
        if context
        else ""
    )
    en_html = (
        f'<div style="margin-top:4px;font-size:11px;color:#94a3b8;line-height:1.5;font-style:italic;">'
        f'"{escape(quote_en)}"</div>'
        if quote_en
        else ""
    )
    return (
        '<div style="border-left:4px solid #FF6B35;padding:14px 18px;background:#fff8f3;'
        'border-radius:0 12px 12px 0;">'
        f'<div style="font-size:16px;line-height:1.55;color:#0f172a;font-weight:600;">'
        f'"{escape(quote_ko)}"</div>'
        f'{en_html}'
        f'<div style="margin-top:10px;font-size:13px;color:#FF6B35;font-weight:700;">'
        f'— {escape(author_ko)} ({escape(author_en)})</div>'
        f'{context_html}'
        '</div>'
    )


def render_quote_block_text(quote_data: dict, context: str) -> str:
    parts = [
        f'  "{quote_data.get("quote_ko", "")}"',
        f'  — {quote_data.get("author_ko", "")} ({quote_data.get("author_en", "")})',
    ]
    if context:
        parts.append(f"  · {context}")
    return "\n".join(parts)


def render_headlines_html(headlines: list[Headline]) -> str:
    if not headlines:
        return (
            '<p style="margin:0;color:#6b7280;font-size:14px;">'
            "현재 표시할 뉴스가 없습니다.</p>"
        )

    items = []
    for h in headlines:
        primary = h.title_ko if h.title_ko else h.title
        secondary_html = (
            f'<div style="margin-top:3px;font-size:12px;color:#94a3b8;line-height:1.4;">'
            f'{escape(h.title)}</div>'
            if h.title_ko and h.title_ko != h.title
            else ""
        )
        source_badge = (
            f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
            f'background:#fff1ea;color:{ACCENT};font-size:10px;font-weight:700;'
            f'letter-spacing:0.02em;">{escape(h.source)}</span>'
            if h.source
            else ""
        )
        published_html = (
            f'<span style="font-size:11px;color:#cbd5e1;margin-left:8px;">{escape(h.published)}</span>'
            if h.published
            else ""
        )
        items.append(
            f"""
            <li style="margin:0 0 14px 0;list-style:none;padding:12px 14px;background:#ffffff;border:1px solid #f1f5f9;border-radius:10px;">
              <div style="margin-bottom:6px;">{source_badge}{published_html}</div>
              <a href="{escape(h.link)}" style="text-decoration:none;color:#0f172a;font-weight:700;font-size:15px;line-height:1.45;">
                {escape(primary)}
              </a>
              {secondary_html}
            </li>
            """
        )

    return f"""
    <ul style="margin:0;padding:0;list-style:none;">
      {''.join(items)}
    </ul>
    """


def render_quotes_text(quotes: list[Quote]) -> str:
    if not quotes:
        return "현재 시장 데이터를 불러올 수 없습니다."
    lines = []
    for q in quotes:
        price = f"${q.price:,.2f}" if q.price is not None else "—"
        if q.change is not None and q.change_pct is not None:
            arrow = "+" if q.is_up else ""
            change = f"{arrow}{q.change:.2f} ({arrow}{q.change_pct:.2f}%) 오늘"
        else:
            change = "— 오늘"
        if q.change_pct_1m is not None:
            arrow_1m = "+" if q.is_up_1m else ""
            month = f"{arrow_1m}{q.change_pct_1m:.2f}% 1개월"
        else:
            month = "— 1개월"
        lines.append(f"  {q.symbol:<6} {price:>10}  {change} / {month}")
        if q.reason:
            lines.append(f"         · {q.reason}")
        if q.article_title and q.article_url:
            lines.append(f"         📰 {q.article_title}")
            lines.append(f"            {q.article_url}")
    return "\n".join(lines)


def render_headlines_text(headlines: list[Headline]) -> str:
    if not headlines:
        return "현재 표시할 뉴스가 없습니다."
    lines = []
    for i, h in enumerate(headlines, 1):
        primary = h.title_ko if h.title_ko else h.title
        lines.append(f"  {i}. {primary}")
        if h.title_ko and h.title_ko != h.title:
            lines.append(f"     ({h.title})")
        if h.source or h.published:
            lines.append(f"     {h.source} · {h.published}")
        lines.append(f"     {h.link}")
    return "\n".join(lines)


KOREAN_WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]


def format_korean_date(dt: datetime) -> str:
    return f"{dt.year}년 {dt.month}월 {dt.day}일 {KOREAN_WEEKDAYS[dt.weekday()]}"


def build_market_context(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    feed_url: str = DEFAULT_NEWS_FEED,
    headline_limit: int = 3,
) -> dict:
    log.info("Fetching market data for %s", ", ".join(tickers))
    quotes = fetch_quotes(tickers)
    log.info("Fetching top %d headlines from %s", headline_limit, feed_url)
    headlines = fetch_headlines(feed_url, headline_limit, fallback_symbols=tickers)

    quote_inputs = [
        {"symbol": q.symbol, "name": q.name, "change_pct": q.change_pct}
        for q in quotes
    ]
    headlines_en = [h.title for h in headlines]

    # Daily investor quote rotates by day-of-year. Lesson is generated daily
    # by the AI based on the actual market situation (no rotation list).
    now = datetime.now()
    quote_data = pick_quote(now)

    summary = ai_summarizer.summarize(
        quote_inputs,
        headlines_en,
        quote_data=quote_data,
    )

    for q in quotes:
        q.reason = summary["reasons"].get(q.symbol, "")
    for h, ko in zip(headlines, summary["headlines_ko"]):
        h.title_ko = ko

    summary_points = summary.get("summary_points") or []
    market_lesson = summary.get("market_lesson") or ""
    quote_context = summary.get("quote_context") or ""

    education_html = render_lesson_html(market_lesson)
    education_text = render_lesson_text(market_lesson)

    quote_html = render_quote_block_html(quote_data, quote_context)
    quote_text = render_quote_block_text(quote_data, quote_context)

    return {
        "market_summary_html": render_summary_html(summary_points),
        "market_summary_text": render_summary_text(summary_points),
        "market_quotes_html": render_quotes_html(quotes),
        "market_quotes_text": render_quotes_text(quotes),
        "market_headlines_html": render_headlines_html(headlines),
        "market_headlines_text": render_headlines_text(headlines),
        "education_html": education_html,
        "education_text": education_text,
        "education_kind": "lesson",
        "quote_html": quote_html,
        "quote_text": quote_text,
        "market_date": format_korean_date(now),
        "market_quote_count": str(len(quotes)),
        "market_headline_count": str(len(headlines)),
        "market_summary_count": str(len(summary_points)),
    }
