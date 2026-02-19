from opentulpa.interfaces.telegram.client import _resolve_media_send_target


def test_resolve_media_send_target_prefers_animation_for_gif_mime() -> None:
    method, field = _resolve_media_send_target(
        kind="photo",
        filename="trend.bin",
        mime_type="image/gif",
    )
    assert method == "sendAnimation"
    assert field == "animation"


def test_resolve_media_send_target_prefers_animation_for_gif_extension() -> None:
    method, field = _resolve_media_send_target(
        kind="photo",
        filename="trend.gif",
        mime_type="application/octet-stream",
    )
    assert method == "sendAnimation"
    assert field == "animation"


def test_resolve_media_send_target_photo_for_standard_images() -> None:
    method, field = _resolve_media_send_target(
        kind="photo",
        filename="trend.jpg",
        mime_type="image/jpeg",
    )
    assert method == "sendPhoto"
    assert field == "photo"
