"""
keep_alive.py — Keeps Hugging Face Space awake by pinging it every 5 minutes.

HOW IT WORKS:
- Runs a background thread that sends an HTTP GET to /ping every 5 minutes
- Hugging Face free-tier spaces sleep after 48h of inactivity
- This self-ping prevents that sleep

USAGE:
- This is automatically imported and started by app.py
- No manual action needed — it runs silently in the background
"""

import threading
import time
import os
import urllib.request
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────
# Ping every 5 minutes (300 seconds)
PING_INTERVAL_SECONDS = 300

# The URL to ping — auto-detects HuggingFace Space URL or falls back to localhost
def _get_ping_url() -> str:
    # Hugging Face sets SPACE_HOST automatically in every Space
    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        return f"https://{space_host}/ping"
    # Fallback: ping localhost (useful for local dev too)
    port = os.environ.get("PORT", "10000")
    return f"http://127.0.0.1:{port}/ping"


# ── Pinger thread ─────────────────────────────────────────────
def _ping_loop():
    url = _get_ping_url()
    logger.info(f"[KeepAlive] Started — pinging {url} every {PING_INTERVAL_SECONDS}s")

    # Wait 30s on first launch before pinging (let server fully start)
    time.sleep(30)

    while True:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "KuldeepAI-KeepAlive/1.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.getcode()
                logger.info(f"[KeepAlive] Ping OK — HTTP {status}")
        except Exception as e:
            logger.warning(f"[KeepAlive] Ping failed: {e}")

        time.sleep(PING_INTERVAL_SECONDS)


# ── Public API ────────────────────────────────────────────────
def start_keep_alive():
    """Start the keep-alive background thread. Call once at app startup."""
    thread = threading.Thread(target=_ping_loop, daemon=True, name="keep-alive")
    thread.start()
    logger.info("[KeepAlive] Background thread started.")
