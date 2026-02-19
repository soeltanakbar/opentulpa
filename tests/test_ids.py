import re

import pytest

from opentulpa.core.ids import new_short_id


def test_new_short_id_format() -> None:
    value = new_short_id("task")
    assert re.match(r"^task_[0-9a-z]{6}$", value)
    assert len(value) == len("task_") + 6


def test_new_short_id_rejects_bad_prefix() -> None:
    with pytest.raises(ValueError):
        new_short_id("9task")
    with pytest.raises(ValueError):
        new_short_id("task!")


def test_new_short_id_is_unique_for_batch() -> None:
    generated = {new_short_id("file") for _ in range(500)}
    assert len(generated) == 500
