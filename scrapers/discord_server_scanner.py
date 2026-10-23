"""
Discord Server Bot Scanner

Logs into Discord via Playwright, searches a channel for bot messages,
and extracts unique bot names across paginated results. Credentials
are loaded from a .env file (never hardcoded).

The output JSON can be fed into the BotLens MCP server for batch
privacy salience report (supports the server-audit use case, paper Sec. V-C2).

NOT REQUIRED FOR ARTIFACT EVALUATION: this script authenticates as a Discord
user, which is contrary to Discord's Terms of Service. scan_server_bots accepts
a plain text file of bot names instead; see ARTIFACT-EVALUATION.md Sec. 5.

Usage:
    python scrapers/discord_server_scanner.py --url "https://discord.com/channels/..." --pages 50
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import asyncio
import json
import os
import random
import argparse
from pathlib import Path

from playwright.async_api import async_playwright
from dotenv import load_dotenv

# Load credentials from .env (never hardcode)
load_dotenv()

DISCORD_LOGIN_URL = "https://discord.com/login"


def human_delay(min_s=0.8, max_s=2.0):
    """Random delay to mimic human behaviour."""
    return random.uniform(min_s, max_s) * 1000


async def login_discord(page, email: str, password: str, timeout: int = 300000):
    """Login to Discord with 2FA support."""
    print("[*] Navigating to Discord login...")
    await page.goto(DISCORD_LOGIN_URL, wait_until="networkidle")

    print("[*] Entering credentials...")
    await page.fill('input[name="email"]', email)
    await page.wait_for_timeout(human_delay(0.5, 1.2))
    await page.fill('input[name="password"]', password)
    await page.wait_for_timeout(human_delay(0.5, 1.2))
    await page.click('button[type="submit"]')

    print("[*] Complete 2FA in the browser window if prompted...")
    await page.wait_for_url("**/app", timeout=timeout)
    print("[+] Login successful!")
    await page.wait_for_timeout(human_delay(2, 4))


async def open_channel(page, discord_url: str):
    """Navigate to a specific Discord server/channel."""
    print(f"[*] Opening channel: {discord_url}")
    await page.goto(discord_url, wait_until="networkidle")
    await page.wait_for_timeout(human_delay(2, 4))


async def search_for_bots(page):
    """Trigger Discord search for authorType: bot."""
    print("[*] Opening search bar...")
    try:
        await page.click("div[class*='search'] div[class*='icon']", timeout=10000)
        await page.wait_for_timeout(human_delay(0.5, 1))
    except Exception:
        pass

    editor = page.locator("div[role='combobox'][contenteditable='true']")
    await editor.click()
    await page.wait_for_timeout(human_delay(0.4, 0.8))

    await editor.press("Control+A")
    await page.wait_for_timeout(human_delay(0.2, 0.4))
    await editor.press("Delete")

    query = "authorType: bot"
    print(f"[*] Searching: {query}")
    for ch in query:
        await editor.type(ch, delay=human_delay(0.03, 0.12))

    await editor.press("Enter")
    print("[*] Search submitted, waiting for results...")
    await page.wait_for_timeout(human_delay(3, 5))


async def extract_bots_with_pagination(page, max_pages: int = 300) -> list[str]:
    """Paginate through search results and collect unique bot names."""
    print(f"\n[*] Paginating through up to {max_pages} pages...")
    bot_names = set()
    page_num = 1

    for page_num in range(1, max_pages + 1):
        print(f"\n--- Page {page_num} ---")
        await page.wait_for_timeout(2000)

        # Scroll within page to load all results
        try:
            scroller = page.locator("div[class*='scroller'][class*='thin']").first
            await scroller.wait_for(state="visible", timeout=5000)
            for _ in range(20):
                await scroller.evaluate("(el) => el.scrollBy(0, 800)")
                await page.wait_for_timeout(human_delay(0.3, 0.6))
            await page.wait_for_timeout(1000)
        except Exception as e:
            print(f"[WARN] Scroll issue on page {page_num}: {e}")

        # Extract bot names
        try:
            badge_containers = page.locator("span[class*='hasBadges']")
            count = await badge_containers.count()
            print(f"[*] {count} badged users on this page")

            for i in range(count):
                try:
                    container = badge_containers.nth(i)
                    app_badge = container.locator("span.botText__82f07")
                    if await app_badge.count() > 0:
                        badge_text = await app_badge.inner_text()
                        if badge_text.strip() == "APP":
                            username_div = container.locator(
                                "div[class*='username'][data-text]"
                            )
                            if await username_div.count() > 0:
                                name = await username_div.get_attribute("data-text")
                                if name and name.strip():
                                    bot_names.add(name.strip())
                except Exception:
                    continue

            print(f"[+] Unique bots so far: {len(bot_names)}")

        except Exception as e:
            print(f"[ERROR] Extraction error on page {page_num}: {e}")

        # Next page
        if page_num < max_pages:
            try:
                next_btn = page.locator("button[rel='next']:not([disabled])")
                if await next_btn.count() > 0:
                    print(f"[>>] Going to page {page_num + 1}...")
                    await next_btn.click()
                    await page.wait_for_timeout(human_delay(2.0, 3.5))
                else:
                    print("[STOP] No more pages.")
                    break
            except Exception as e:
                print(f"[WARN] Pagination error: {e}")
                break

    print(f"\n[+] Done. Scanned {page_num} page(s), found {len(bot_names)} unique bots.")
    return sorted(bot_names)


def save_results(bot_names: list[str], output_path: str = "discord_bots_found.json"):
    """Save bot names to JSON for BotLens MCP consumption."""
    data = {
        "total_bots": len(bot_names),
        "bots": bot_names,
        # BotLens MCP will read this file via scan_server_bots tool
        "_botguard_note": "Feed this file to BotLens MCP scan_server_bots tool for RQ5 analysis"
    }
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[+] Saved {len(bot_names)} bots to: {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(description="Discord Bot Scanner for BotLens RQ5")
    parser.add_argument("--url",    required=True,  help="Discord channel URL to scan")
    parser.add_argument("--pages",  type=int, default=300, help="Max pages to scan (default 300)")
    parser.add_argument("--output", default="discord_bots_found.json", help="Output JSON file")
    args = parser.parse_args()

    # Load credentials from .env
    email    = os.getenv("DISCORD_EMAIL")
    password = os.getenv("DISCORD_PASSWORD")

    if not email or not password:
        print("[!] ERROR: DISCORD_EMAIL and DISCORD_PASSWORD must be set in .env file")
        print("    Copy .env.example to .env and fill in your credentials.")
        return

    print("=== BotLens Discord Server Scanner ===")
    print(f"    Channel : {args.url}")
    print(f"    Max pages: {args.pages}")
    print(f"    Output  : {args.output}")
    print()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context()
        page    = await context.new_page()

        try:
            await login_discord(page, email, password)
            await open_channel(page, args.url)
            await search_for_bots(page)
            bot_names = await extract_bots_with_pagination(page, max_pages=args.pages)

            print("\n=== Bots Found ===")
            for bot in bot_names:
                print(f"  [BOT] {bot}")

            output_file = save_results(bot_names, args.output)

            print("\n=== Next Step: RQ5 Analysis ===")
            print("In Claude Desktop, run:")
            print(f'  "Scan server bots from {output_file} and give me a full privacy risk report"')
            if bot_names:
                print(f'\n  Or check individual bots:')
                for b in bot_names[:3]:
                    print(f'  "Check safety of {b}"')

            await page.wait_for_timeout(5000)

        except Exception as e:
            print(f"\n[ERROR] {e}")
            await page.wait_for_timeout(30000)
        finally:
            await browser.close()
            print("[DONE] Browser closed.")


if __name__ == "__main__":
    asyncio.run(main())