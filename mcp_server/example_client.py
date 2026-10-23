"""
Example BotLens MCP client (paper Sec. V-C, Use Cases 1 and 2)
==============================================================
A minimal client built on the official MCP Python SDK. It starts the BotLens
server as a subprocess over stdio, exactly as Claude Desktop or any other MCP
client would, lists the six tools, and reproduces both use cases:

    Use Case 1 (Fig. 15)  get_bot_signals("Wick")
    Use Case 2 (Fig. 16)  scan_server_bots(<manifest of the six Fig. 16 bots>)

No LLM, API key or network access is needed.

Usage (from the repository root):
    python mcp_server/example_client.py
    python mcp_server/example_client.py --tool get_taxonomy_breakdown --arg bot_name=Wick
"""

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER = REPO_ROOT / "mcp_server" / "server.py"

# The six bots of Fig. 16. Fig. 16 prints the fifth as "Dank Member"; the bot is
# "Dank Memer" on Top.gg and in the dataset.
FIG16_BOTS = ["Wick", "Auttaja", "Carl-bot", "Dyno", "Dank Memer", "Mudae"]


def _print_result(title: str, result) -> None:
    print(f"\n{'#' * 72}\n# {title}\n{'#' * 72}")
    for block in result.content:
        print(getattr(block, "text", block))
    if result.isError:
        print("(tool reported an error)")


async def run(tool: str | None, args: dict) -> None:
    env = dict(os.environ)
    env.setdefault("BOTLENS_DATA_DIR", str(REPO_ROOT))
    env["PYTHONIOENCODING"] = "utf-8"
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=env)

    print(f"Starting BotLens server: {SERVER}")
    print(f"BOTLENS_DATA_DIR = {env['BOTLENS_DATA_DIR']}  (loading takes ~10 s)")

    # The server logs per-call timings to stderr; keep the console readable.
    with open(os.devnull, "w") as devnull:
        async with stdio_client(params, errlog=devnull) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = (await session.list_tools()).tools
                print(f"\nServer exposes {len(tools)} tools:")
                for t in tools:
                    print(f"  - {t.name}")

                if tool:
                    _print_result(f"{tool}({args})", await session.call_tool(tool, args))
                    return

                _print_result(
                    "Use Case 1 (Fig. 15): get_bot_signals('Wick')",
                    await session.call_tool("get_bot_signals", {"bot_name": "Wick"}),
                )

                with tempfile.TemporaryDirectory() as tmp:
                    manifest = Path(tmp) / "bots.txt"
                    manifest.write_text("\n".join(FIG16_BOTS) + "\n", encoding="utf-8")
                    _print_result(
                        "Use Case 2 (Fig. 16): scan_server_bots(<six-bot manifest>)",
                        await session.call_tool("scan_server_bots", {"filepath": str(manifest)}),
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tool", help="call a single tool instead of the two use cases")
    parser.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE",
                        help="tool argument; repeat for several (numbers are converted)")
    ns = parser.parse_args()

    args = {}
    for item in ns.arg:
        key, _, value = item.partition("=")
        args[key] = int(value) if value.isdigit() else value
    asyncio.run(run(ns.tool, args))


if __name__ == "__main__":
    main()
