"""BotLens configuration package.

Makes stdout/stderr UTF-8 tolerant on import. Bot names and review text contain
emoji and non-Latin scripts; Windows consoles default to cp1252, which cannot
encode them. Every pipeline script imports config.settings, so this covers them all.
"""
import sys

for _stream in (sys.stdout, sys.stderr):
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
