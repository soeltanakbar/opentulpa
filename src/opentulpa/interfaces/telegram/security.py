"""Telegram access-control helpers."""

from __future__ import annotations


def parse_csv_set(value: str | None, *, normalize_username: bool = False) -> set[str]:
    if not value:
        return set()
    out: set[str] = set()
    for part in value.split(","):
        v = part.strip()
        if not v:
            continue
        if normalize_username:
            if v.startswith("@"):
                v = v[1:]
            v = v.lower()
        out.add(v)
    return out


def is_user_allowed(
    *,
    user_id: int,
    username: str | None,
    allowed_user_ids_csv: str | None,
    allowed_usernames_csv: str | None,
) -> bool:
    allowed_ids = parse_csv_set(allowed_user_ids_csv)
    allowed_usernames = parse_csv_set(allowed_usernames_csv, normalize_username=True)
    if not allowed_ids and not allowed_usernames:
        return True
    if str(user_id) in allowed_ids:
        return True
    return bool(username and username.lower() in allowed_usernames)
