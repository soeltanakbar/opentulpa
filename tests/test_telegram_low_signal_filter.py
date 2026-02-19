from opentulpa.interfaces.telegram.relay import is_low_signal_reply


def test_punctuation_only_reply_is_low_signal() -> None:
    assert is_low_signal_reply("...")
    assert is_low_signal_reply("..")
    assert is_low_signal_reply("!")


def test_normal_reply_is_not_low_signal() -> None:
    assert not is_low_signal_reply("I found 3 relevant items.")
