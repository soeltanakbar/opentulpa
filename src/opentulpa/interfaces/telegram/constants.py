"""Telegram interface constants and filesystem paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STATE_PATH = PROJECT_ROOT / ".opentulpa" / "telegram_state.json"
ENV_PATH = PROJECT_ROOT / ".env"
DEBUG_LOG_PATH = PROJECT_ROOT / ".cursor" / "debug.log"

BLOCKED_ENV_KEYS = {
    "PATH",
    "HOME",
    "PWD",
    "SHELL",
    "USER",
    "LOGNAME",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
}

LOW_SIGNAL_REPLIES = {
    "i see",
    "understood",
    "let me see",
    "checking this",
    "checking",
    "working on it",
    "acknowledged",
}
