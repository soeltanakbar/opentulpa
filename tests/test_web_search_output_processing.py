from __future__ import annotations

from opentulpa.integrations.web_search import _extract_sources, _sanitize_answer_text


def test_sanitize_answer_text_removes_favicon_noise() -> None:
    raw = (
        "Key points here.\n"
        "Favicon for https://example.com/article\n"
        "Another useful line.\n"
        "Previous slideNext slide\n"
    )
    cleaned = _sanitize_answer_text(raw)
    assert "Favicon for" not in cleaned
    assert "Previous slideNext slide" not in cleaned
    assert "Key points here." in cleaned
    assert "Another useful line." in cleaned


def test_extract_sources_collects_from_payload_and_answer() -> None:
    data = {
        "citations": [
            {"url": "https://one.example/a"},
            "https://two.example/b",
        ],
        "choices": [
            {
                "message": {
                    "sources": [{"link": "https://three.example/c"}],
                    "content": "See https://four.example/d for more.",
                }
            }
        ],
    }
    answer = "Summary with link https://four.example/d and https://two.example/b"
    sources = _extract_sources(data, answer)
    urls = [item["url"] for item in sources]
    assert "https://one.example/a" in urls
    assert "https://two.example/b" in urls
    assert "https://three.example/c" in urls
    assert "https://four.example/d" in urls
    # de-dup
    assert urls.count("https://two.example/b") == 1

