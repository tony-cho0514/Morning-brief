"""LLM-powered translation + per-stock reason generation.

Uses the Replit AI Integrations OpenAI proxy. If the proxy env vars are not
configured or the call fails for any reason, helpers degrade gracefully: the
caller receives empty strings and the rest of the newsletter still renders.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Sequence

log = logging.getLogger("newsletter.ai")

_MODEL = os.getenv("NEWSLETTER_AI_MODEL", "gpt-5.4")


def _client():
    base_url = os.getenv("AI_INTEGRATIONS_OPENAI_BASE_URL")
    api_key = os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY")
    if not base_url or not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed; skipping AI step")
        return None
    return OpenAI(base_url=base_url, api_key=api_key)


def summarize(
    quote_inputs: Sequence[dict],
    headlines_en: Sequence[str],
    *,
    quote_data: dict | None = None,
) -> dict:
    """Return AI-generated Korean copy for the newsletter.

    quote_inputs: [{"symbol": "NVDA", "name": "NVIDIA", "change_pct": 4.12}, ...]
    headlines_en: list of English headline strings (top N).
    quote_data:   {"author_ko": "워런 버핏", "quote_ko": "..."} — when provided,
                  AI generates a 2-sentence "why it matters today" context.

    Reasons are generated as analyst-style Korean paragraphs (100+ chars each).
    A market_lesson tied to today's actual situation is always generated.

    Always returns: reasons, headlines_ko, summary_points, market_lesson,
    quote_context. Missing/optional fields come back as empty strings.
    """
    fallback = {
        "reasons": {q["symbol"]: "" for q in quote_inputs},
        "headlines_ko": [""] * len(headlines_en),
        "summary_points": [],
        "market_lesson": "",
        "quote_context": "",
    }

    client = _client()
    if client is None:
        log.info("AI proxy not configured; skipping translation/reasons")
        return fallback

    quote_lines = "\n".join(
        f"- {q['symbol']} ({q.get('name') or q['symbol']}): "
        f"{'전일대비 ' + format(q['change_pct'], '+.2f') + '%' if q.get('change_pct') is not None else '변동 없음'}"
        for q in quote_inputs
    )
    headline_lines = "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines_en))

    extra_tasks = ""
    extra_json_lines = ""
    if quote_data:
        extra_tasks += (
            f'\n5) {quote_data["author_ko"]}의 명언 "{quote_data["quote_ko"]}" 이(가) '
            "오늘의 시장 데이터·헤드라인 맥락에서 왜 의미 있는지 한국어 2문장으로 설명하세요. "
            '"~예요", "~이에요" 같은 부드러운 종결을 사용하고, 너무 교훈적이지 않게.'
        )
        extra_json_lines += ',\n  "quote_context": "<오늘의 시장과 연결한 한국어 2문장>"'

    user_prompt = f"""다음은 오늘 미국 주식 시장 데이터입니다.

[종목 등락]
{quote_lines}

[영문 헤드라인]
{headline_lines if headline_lines else '(없음)'}

작업:
1) 각 종목의 오늘 움직임을 분석가가 클라이언트에게 보내는 데일리 리포트처럼 작성하세요.
   - **반드시 한국어 100자 이상**, 한 문단으로 자연스럽게 이어지게.
   - 다음 4가지를 모두 포함하세요:
     (가) 오늘 어떤 움직임이 있었나 (등락의 사실 묘사)
     (나) 왜 그랬을 가능성이 큰가 (헤드라인 또는 거시·섹터 흐름과의 연결)
     (다) 시장 맥락 (실적 시즌·금리·AI 사이클·경쟁 구도 등 배경 한 가지)
     (라) 앞으로 봐야 할 포인트 (다음 이벤트, 지표, 가격대 등)
   - 마침표 사용 가능. 종결은 "~예요", "~이에요", "~인 셈이에요", "~봐야 해요" 같은 친근한 어투.
   - 단정적 예측("반드시 오를 것"류)은 금지, 가능성 표현 사용.
2) 각 영문 헤드라인을 자연스러운 한국어로 번역하세요. 25자 내외, 핵심만 압축한 명사구 형태로.
3) "오늘의 핵심 3가지" — 오늘 시장 전반을 요약하는 한국어 불릿 3개를 작성하세요.
   - 각 불릿은 15~28자, 명사구 또는 짧은 문장. 마침표 금지.
   - 종목 등락과 헤드라인을 종합해 가장 임팩트 있는 3가지를 선정.
   - 좋은 예: "코스피 사상 최고치 경신", "NVDA AI 수요로 4% 급등", "연준 금리 동결 시사로 안도 랠리".
4) "오늘의 시장 인사이트 (market_lesson)" — 위 데이터를 바탕으로 일반 투자자에게 도움되는 실용적인 교훈을 한국어 150~200자로 작성하세요.
   - **단순 시황 요약 금지.** 오늘 시장 상황을 빌려 알려주는 **투자 원리/개념/심리 한 가지**여야 해요.
   - 예시 방향:
     · 강한 랠리 후 조정이라면 → "왜 강한 상승 뒤에는 종종 숨고르기가 따라오는지, 똑똑한 투자자들이 이때 무엇을 하는지"
     · AI 종목이 떨어진다면 → "실적 시즌이 무엇이고 어닝 결과를 어떻게 읽는지"
     · 한 종목만 튀었다면 → "분산투자가 왜 중요한지를 오늘 SPY 대비 NVDA 격차로 설명"
   - 친근한 어투(~예요/이에요), 끝에 한 줄 행동지침이나 마인드셋 포함.
   - 글머리표 없이 자연스러운 단락 한두 개로.{extra_tasks}

다음 JSON만 출력하세요. 다른 텍스트는 절대 포함하지 마세요:
{{
  "reasons": {{ "<SYMBOL>": "<한국어 100자 이상 분석 한 단락>", ... }},
  "headlines_ko": ["<번역1>", "<번역2>", ...],
  "summary_points": ["<핵심1>", "<핵심2>", "<핵심3>"],
  "market_lesson": "<오늘의 데이터를 빌린 150~200자 한국어 투자 교훈>"{extra_json_lines}
}}"""

    try:
        resp = client.chat.completions.create(
            model=_MODEL,
            max_completion_tokens=3800,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Korean financial newsletter editor. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
    except Exception as exc:
        log.warning("AI summarize failed: %s", exc)
        return fallback

    reasons_raw = data.get("reasons") or {}
    headlines_ko_raw = data.get("headlines_ko") or []
    summary_points_raw = data.get("summary_points") or []

    reasons = {q["symbol"]: "" for q in quote_inputs}
    if isinstance(reasons_raw, dict):
        for sym, txt in reasons_raw.items():
            if isinstance(txt, str) and sym in reasons:
                reasons[sym] = txt.strip()

    headlines_ko: list[str] = []
    for i in range(len(headlines_en)):
        if i < len(headlines_ko_raw) and isinstance(headlines_ko_raw[i], str):
            headlines_ko.append(headlines_ko_raw[i].strip())
        else:
            headlines_ko.append("")

    summary_points: list[str] = []
    if isinstance(summary_points_raw, list):
        for item in summary_points_raw:
            if isinstance(item, str) and item.strip():
                summary_points.append(item.strip())
    summary_points = summary_points[:3]

    market_lesson = data.get("market_lesson") or ""
    if not isinstance(market_lesson, str):
        market_lesson = ""
    market_lesson = market_lesson.strip()

    quote_context = data.get("quote_context") or ""
    if not isinstance(quote_context, str):
        quote_context = ""
    quote_context = quote_context.strip()

    avg_reason_len = (
        int(sum(len(v) for v in reasons.values()) / max(1, sum(1 for v in reasons.values() if v)))
        if any(reasons.values())
        else 0
    )
    log.info(
        "AI summarize ok: %d reasons (avg %d chars), %d translated headlines, "
        "%d summary points, lesson=%d chars, quote_ctx=%d chars",
        sum(1 for v in reasons.values() if v),
        avg_reason_len,
        sum(1 for v in headlines_ko if v),
        len(summary_points),
        len(market_lesson),
        len(quote_context),
    )
    return {
        "reasons": reasons,
        "headlines_ko": headlines_ko,
        "summary_points": summary_points,
        "market_lesson": market_lesson,
        "quote_context": quote_context,
    }
