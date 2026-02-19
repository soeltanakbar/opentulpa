from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_strip_internal_selected_prefix() -> None:
    text = '{"selected": []}Hello! How can I help?'
    cleaned = OpenTulpaLangGraphRuntime._strip_internal_json_prefix(text)
    assert cleaned == "Hello! How can I help?"


def test_strip_internal_classifier_prefix() -> None:
    text = '{"notify_user": false, "reason": "low priority"}Done.'
    cleaned = OpenTulpaLangGraphRuntime._strip_internal_json_prefix(text)
    assert cleaned == "Done."


def test_do_not_strip_user_facing_json() -> None:
    text = '{"weather":"rain","temp_c":27}'
    cleaned = OpenTulpaLangGraphRuntime._strip_internal_json_prefix(text)
    assert cleaned == text


def test_detect_incomplete_internal_prefix_from_selector() -> None:
    partial = '{"selected": [{"name":"skill-creator","score":1.0'
    assert OpenTulpaLangGraphRuntime._strip_internal_json_prefix(partial) == partial
    assert OpenTulpaLangGraphRuntime._has_incomplete_internal_json_prefix(partial) is True


def test_do_not_flag_incomplete_non_internal_json() -> None:
    partial = '{"weather":"rai'
    assert OpenTulpaLangGraphRuntime._has_incomplete_internal_json_prefix(partial) is False


def test_extract_json_object_from_wrapped_text() -> None:
    raw = 'prefix {"notify_user": true, "reason": "urgent"} suffix'
    parsed = OpenTulpaLangGraphRuntime._extract_json_object(raw)
    assert parsed == {"notify_user": True, "reason": "urgent"}
