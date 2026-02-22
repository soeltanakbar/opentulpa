import pytest

from opentulpa.api.file_helpers import (
    download_image_from_web_url,
    infer_image_filename,
    safe_telegram_filename,
)


def test_safe_telegram_filename_sanitizes() -> None:
    assert safe_telegram_filename("weird name?.png") == "weird_name_.png"


def test_infer_image_filename_adds_extension() -> None:
    name = infer_image_filename("https://example.com/path/cat", "image/png")
    assert name.endswith(".png")


@pytest.mark.asyncio
async def test_download_image_rejects_non_http_scheme() -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        await download_image_from_web_url("ftp://example.com/image.png")
