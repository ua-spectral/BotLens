"""
BotLens tool latency (paper Sec. V-D)
=====================================
Replays a session with the same composition as the paper's 22-query session:
11 x get_bot_signals, 1 x get_taxonomy_breakdown, 4 x get_review_samples and
6 x scan_server_bots, then prints per-tool mean / min / max from the server's
own timer (get_timing_stats). Dataset loading at startup (~10 s) is one-time
cost and is not included, as in the paper.

Usage (from the repository root):
    python mcp_server/benchmark_latency.py
"""

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("BOTLENS_DATA_DIR", str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mcp_server"))

import server  # noqa: E402  (loads the dataset)

if not server._DATA_LOADED:
    sys.exit(f"Dataset not loaded:\n{server._LOAD_ERROR}")

BOTS = ["Wick", "Auttaja", "Carl-bot", "Dyno", "Dank Memer", "Mudae"]

# The tools log each call to stderr; silence that while the session runs.
_stderr, sys.stderr = sys.stderr, open(os.devnull, "w")
try:
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "bots.txt"
        manifest.write_text("\n".join(BOTS) + "\n", encoding="utf-8")

        for i in range(11):
            server.get_bot_signals(BOTS[i % len(BOTS)])
        server.get_taxonomy_breakdown("Wick")
        for name in BOTS[:4]:
            server.get_review_samples(name)
        for _ in range(6):
            server.scan_server_bots(str(manifest))
finally:
    sys.stderr.close()
    sys.stderr = _stderr

server._TIMING_LOG.pop("get_timing_stats", None)
print(server.get_timing_stats())
