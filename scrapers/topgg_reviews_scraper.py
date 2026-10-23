"""
Top.gg Review Scraper — Async Parallel

Scrapes user reviews from Top.gg bot pages using Playwright.
Runs multiple browser tabs concurrently for throughput, handles
pagination and deduplication, and saves one JSON file per bot.

Usage:
    python scrapers/topgg_reviews_scraper.py

Input:  unique_bot_ids.txt  (one bot ID per line)
Output: bot_reviews_json/<bot_id>.json
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import json
import asyncio
import os
import sys
from typing import List 
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("ERROR: Playwright not installed!")
    sys.exit(1)

# --- CONFIGURATION ---
INPUT_FILE = "unique_bot_ids.txt"
OUTPUT_DIR = "bot_reviews_json"
CONCURRENT_LIMIT = 4  # Number of bots to scrape at the same time

@dataclass
class Review:
    # NOTE: Reviewer display names are visible on the public Top.gg page but are
    # deliberately NOT retained. We store only the review content and public
    # metadata below; reviewer identifiers are not collected into the dataset.
    rating: int
    content: str
    timestamp: str
    upvotes: int

@dataclass
class BotData:
    bot_id: str
    name: str
    reviews: List[Review]
    scraped_at: str

def clean_text(text: str) -> str:
    if not text: return ""
    return " ".join(text.split()).strip()

async def scrape_single_bot(bot_id: str, context, semaphore):
    # This semaphore limits how many run at once
    async with semaphore:
        filename = os.path.join(OUTPUT_DIR, f"{bot_id}.json")
        
        if os.path.exists(filename):
            print(f"[SKIPPING] {bot_id} (Already Done)")
            return

        page = await context.new_page()
        url = f"https://top.gg/bot/{bot_id}"
        
        try:
            print(f"[STARTING] {bot_id}...")
            # Fast load
            await page.goto(url, wait_until='domcontentloaded', timeout=45000)
            
            # 1. NAME
            name = "Unknown"
            name_el = page.locator('h1').first
            if await name_el.is_visible():
                name = await name_el.text_content()
                name = name.strip()

            # 2. REVIEWS LOOP
            # Scroll down
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.7)")
            await asyncio.sleep(2) 
            
            reviews = []
            current_page = 1
            
            while True:
                count_before = len(reviews)
                
                # Scrape
                articles = await page.locator('article').all()
                for art in articles:
                    try:
                        # Reviewer display name is visible here but intentionally
                        # not read or retained. We require a review body to exist,
                        # then capture only the content and public metadata.
                        p_el = art.locator('p.wrap-break-word')
                        content = clean_text(await p_el.text_content())
                        if not content:
                            continue

                        rating = 0
                        stars = art.locator('[aria-label$="stars"]').first
                        if await stars.is_visible():
                            aria = await stars.get_attribute('aria-label')
                            rating = int(aria.split()[0])

                        timestamp = ""
                        # Simple timestamp grab
                        spans = await art.locator('span').all()
                        for span in spans:
                            txt = await span.text_content()
                            if "ago" in txt:
                                timestamp = txt.strip(); break

                        upvotes = 0

                        # Deduplicate on content (and rating) only, no identifier needed
                        if not any(r.content == content and r.rating == rating for r in reviews):
                            reviews.append(Review(rating, content, timestamp, upvotes))
                    except: continue
                
                # Check for new data
                new_found = len(reviews) - count_before
                
                # Pagination
                next_btn = page.locator('button.chakra-button.css-7rul47').last
                
                if await next_btn.is_visible() and not await next_btn.is_disabled():
                    # Stop if loop detected
                    if current_page > 1 and new_found == 0:
                        break
                        
                    await next_btn.click()
                    await asyncio.sleep(1)
                    current_page += 1
                else:
                    break

            # SAVE
            if reviews:
                data = BotData(bot_id, name, reviews, datetime.now().isoformat())
                out = asdict(data)
                out['reviews'] = [asdict(r) for r in data.reviews]
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(out, f, indent=2, ensure_ascii=False)
                print(f"[COMPLETED] {bot_id} -> {len(reviews)} reviews saved.")
            else:
                 print(f"[EMPTY] {bot_id} -> No reviews found.")

        except Exception as e:
            print(f"[ERROR] {bot_id}: {e}")
        finally:
            await page.close()

async def main():
    if not os.path.exists(INPUT_FILE):
        print("Input file missing.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(INPUT_FILE, 'r') as f:
        bot_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(bot_ids)} IDs. Starting Parallel Scrape (Limit: {CONCURRENT_LIMIT})...")

    async with async_playwright() as p:
        # Launch ONE browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent="Mozilla/5.0")
        
        # Limit concurrency
        semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
        
        # Create tasks
        tasks = [scrape_single_bot(bid, context, semaphore) for bid in bot_ids]
        
        # Run all tasks
        await asyncio.gather(*tasks)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())