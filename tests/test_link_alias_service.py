from __future__ import annotations

from pathlib import Path

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.context.link_aliases import LinkAliasService


def _service(tmp_path: Path) -> LinkAliasService:
    return LinkAliasService(db_path=tmp_path / "link_aliases.db")


def test_register_links_from_text_deduplicates_and_trims(tmp_path: Path) -> None:
    service = _service(tmp_path)
    text = (
        "Check this URL: https://Example.com/path?q=1, and again "
        "https://example.com/path?q=1. Also https://docs.example.com/a(b)c)."
    )
    rows = service.register_links_from_text("telegram_1", text, source="user_turn")
    assert len(rows) == 2
    urls = {row["url"] for row in rows}
    assert "https://example.com/path?q=1" in urls
    assert "https://docs.example.com/a(b)c" in urls


def test_expand_link_ids_in_text(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = service.register_link(
        "telegram_1",
        "https://example.com/very/long/link/with/query?foo=bar&x=1",
        source="tool:web_search",
    )
    assert row is not None
    link_id = row["id"]
    expanded = service.expand_link_ids_in_text(
        "telegram_1",
        f"Use {link_id} to open the page.",
    )
    assert "https://example.com/very/long/link/with/query?foo=bar&x=1" in expanded
    assert link_id not in expanded


def test_runtime_resolves_link_ids_in_tool_args(tmp_path: Path) -> None:
    service = _service(tmp_path)
    row = service.register_link(
        "telegram_2",
        "https://openrouter.ai/docs/guides/overview/multimodal/audio",
        source="assistant_turn",
    )
    assert row is not None
    link_id = row["id"]

    runtime = OpenTulpaLangGraphRuntime.__new__(OpenTulpaLangGraphRuntime)
    runtime._link_alias_service = service

    resolved = runtime.resolve_link_aliases_in_args(
        customer_id="telegram_2",
        args={
            "url": link_id,
            "notes": f"see {link_id}",
            "nested": {"items": [link_id]},
        },
    )
    expected = "https://openrouter.ai/docs/guides/overview/multimodal/audio"
    assert resolved["url"] == expected
    assert expected in resolved["notes"]
    assert resolved["nested"]["items"][0] == expected
