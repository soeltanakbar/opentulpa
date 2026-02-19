"""
Web search via OpenRouter's web search feature.

:online is enabled only when the agent explicitly calls the web_search tool
(i.e. when it detects intent or need for current/web information). The main
Parlant agent model never uses :online—only this path does. See:
https://openrouter.ai/announcements/introducing-web-search-via-the-api
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


def _default_online_model() -> str:
    """OpenRouter model with :online suffix for web search."""
    base = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o")
    base = base.rstrip("/").replace(":online", "").strip()
    return f"{base}:online"


async def web_search(query: str, model: str | None = None) -> str:
    """
    Run a web-backed completion: OpenRouter fetches web results and the model
    incorporates them into a single reply. Returns the model's response text.
    """
    api_key = OPENROUTER_API_KEY
    if not api_key:
        return "Web search is not configured (OPENROUTER_API_KEY missing)."

    use_model = (model or _default_online_model()).replace(":online", "").strip() + ":online"
    url = f"{OPENROUTER_BASE}/chat/completions"

    # Optional: pass plugin explicitly. :online in model name is equivalent.
    payload = {
        "model": use_model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 2048,
        "reasoning": {
            "enabled": True,
            "effort": "medium",
            "exclude": True,
        },
    }
    # Use :online only here (agent decided web search was needed); main agent stays offline

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPStatusError as e:
            logger.exception("OpenRouter web search HTTP error: %s", e)
            return f"Web search request failed: {e.response.status_code}."
        except Exception as e:
            logger.exception("OpenRouter web search error: %s", e)
            return f"Web search failed: {e!s}."

    choices = data.get("choices") or []
    if not choices:
        return "No response from web search."
    content = choices[0].get("message", {}).get("content") or ""
    return content.strip() or "No content in response."
