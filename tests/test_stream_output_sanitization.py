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
