"""
Top.gg Bot Metadata Scraper

Collects bot page HTML from Top.gg using parallel undetected-chromedriver
instances. Iterates through category tags to discover bot URLs, then
downloads each page with rendered JavaScript. Supports auto-restart
cycles and resume from saved progress.

Usage:
    python scrapers/topgg_metadata_scraper.py --workers 3
    python scrapers/topgg_metadata_scraper.py --resume --auto-restart 1.0
"""

# Windows consoles default to cp1252; bot names and review text are not.
import sys as _sys

for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


import time
import random
import os
import json
import re
import logging
import argparse
from typing import Set, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import queue

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from tqdm import tqdm


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ScraperConfig:
    """Configuration for the scraper."""
    max_bots: int = 40000
    scroll_pause: Tuple[float, float] = (1.0, 2.0)
    output_dir: str = "bot_pages"
    max_stagnant_rounds: int = 6
    cloudflare_timeout: int = 120
    page_load_timeout: int = 20
    progress_file: str = "scraper_progress.json"
    log_file: str = "scraper.log"
    download_workers: int = 3  # Number of parallel browsers (3-5 recommended)
    batch_size_per_worker: int = 100  # Restart browser after N pages


TAGS = [
    "fun", "moderation", "utility", "music", "economy", "social", "game", "meme",
    "leveling", "anime", "logging", "role-management", "discord-music-bot", "turkish",
    "roleplay", "24-7-music", "games", "customizable-behavior", "gaming",
    "automoderation", "multifunctional", "media", "bot",
    "ticket-system", "minecraft", "multipurpose", "web-dashboard", "moderation-bot",
    "giveaway", "music-bots", "community", "funny", "slash-commands", "mini-games",
    "crypto", "security", "musica", "information", "youtube", "advanced-economy",
    "giveaways", "free", "ai", "entertainment", "administration", "welcomer", "reddit",
    "pokemon", "easy-to-use", "spotify", "artificial-intelligence", "rpg", "antiraid",
    "memes", "server-management", "chat-bot", "roblox", "reaction-roles", "discord",
    "autoroles", "bot-moderation", "fortnite", "moderasyon", "stream", "images",
    "ai-chatbot", "antinuke", "level-rank-leaderboard", "verification", "protection",
    "cryptocurrency", "english", "tickets", "economia", "eglence", "radio", "tools",
    "funbot", "twitch", "ticket", "image-generation", "24-7", "league-of-legends",
    "stats", "minigames", "looking-for-game", "automation", "economy-bot", "chatbot",
    "soundboard", "chat", "chatting", "temporary-voice-channels", "diversão",
    "antispam", "gambling", "logs", "support", "bots", "currency"
]


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file: str, verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger('topgg_scraper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def human_sleep(a: float = 1.0, b: float = 2.0):
    """Sleep for a random duration to simulate human behavior."""
    time.sleep(random.uniform(a, b))


def is_valid_bot_url(url: str) -> bool:
    """Validate that URL is a proper top.gg bot page."""
    pattern = r'^https://top\.gg/bot/\d{17,19}$'
    return bool(re.match(pattern, url))


def clean_bot_url(href: str) -> Optional[str]:
    """Extract and validate bot URL from href attribute."""
    if not href or "/bot/" not in href or "/server/" in href:
        return None
    
    cleaned = href.split("?")[0].rstrip('/')
    
    if is_valid_bot_url(cleaned):
        return cleaned
    return None


# ============================================================================
# PROGRESS MANAGEMENT
# ============================================================================

class ProgressManager:
    """Manages scraping progress and state persistence."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = logging.getLogger('topgg_scraper')
        self.progress_file = Path(config.progress_file)
    
    def save_progress(self, urls: Set[str], tags_completed: List[str]):
        """Save current progress to file."""
        try:
            progress_data = {
                'urls': list(urls),
                'tags_completed': tags_completed,
                'timestamp': time.time(),
                'total_bots': len(urls)
            }
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            self.logger.info(f"Progress saved: {len(urls)} URLs, {len(tags_completed)} tags completed")
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def load_progress(self) -> Tuple[Set[str], List[str]]:
        """Load progress from file if it exists."""
        if not self.progress_file.exists():
            self.logger.info("No previous progress found, starting fresh.")
            return set(), []
        
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            urls = set(data.get('urls', []))
            tags_completed = data.get('tags_completed', [])
            
            self.logger.info(f"Loaded progress: {len(urls)} URLs, {len(tags_completed)} tags completed")
            return urls, tags_completed
        except Exception as e:
            self.logger.error(f"Failed to load progress: {e}")
            return set(), []
    
    def load_existing_urls(self) -> Set[str]:
        """Load URLs from already-scraped HTML files."""
        existing_urls = set()
        output_path = Path(self.config.output_dir)
        
        if not output_path.is_dir():
            return existing_urls
        
        self.logger.info(f"Checking {self.config.output_dir} for existing bot pages...")
        
        for filepath in output_path.glob("*.html"):
            file_base = filepath.stem
            if file_base.isdigit() and len(file_base) >= 17:
                existing_urls.add(f"https://top.gg/bot/{file_base}")
        
        self.logger.info(f"Found {len(existing_urls)} existing bot pages")
        return existing_urls


# ============================================================================
# BROWSER POOL MANAGEMENT
# ============================================================================

class BrowserPool:
    """Manages a pool of undetected_chromedriver instances."""
    
    def __init__(self, pool_size: int, config: ScraperConfig):
        self.pool_size = pool_size
        self.config = config
        self.logger = logging.getLogger('topgg_scraper')
        self.drivers = []
        
    def create_driver(self) -> uc.Chrome:
        """Create a new Chrome driver instance."""
        options = uc.ChromeOptions()
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        
        driver = uc.Chrome(options=options)
        driver.set_page_load_timeout(self.config.page_load_timeout)
        return driver
    
    def initialize(self):
        """Initialize the browser pool."""
        self.logger.info(f"Initializing browser pool with {self.pool_size} instances...")
        for i in range(self.pool_size):
            try:
                driver = self.create_driver()
                self.drivers.append(driver)
                self.logger.info(f"Browser {i+1}/{self.pool_size} initialized")
                time.sleep(2)  # Stagger browser launches
            except Exception as e:
                self.logger.error(f"Failed to create browser {i+1}: {e}")
        
        self.logger.info(f"Browser pool ready: {len(self.drivers)}/{self.pool_size} browsers active")
    
    def get_driver(self, index: int) -> Optional[uc.Chrome]:
        """Get a driver from the pool by index."""
        if 0 <= index < len(self.drivers):
            return self.drivers[index]
        return None
    
    def restart_driver(self, index: int) -> bool:
        """Restart a specific driver."""
        try:
            if 0 <= index < len(self.drivers):
                self.logger.info(f"Restarting browser {index+1}...")
                old_driver = self.drivers[index]
                try:
                    old_driver.quit()
                except:
                    pass
                
                time.sleep(2)
                new_driver = self.create_driver()
                self.drivers[index] = new_driver
                self.logger.info(f"Browser {index+1} restarted successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to restart browser {index+1}: {e}")
            return False
    
    def cleanup(self):
        """Close all browsers."""
        self.logger.info("Closing all browsers...")
        for i, driver in enumerate(self.drivers):
            try:
                driver.quit()
                self.logger.info(f"Browser {i+1} closed")
            except Exception as e:
                self.logger.warning(f"Error closing browser {i+1}: {e}")
        self.drivers = []


# ============================================================================
# BOT URL COLLECTION (SINGLE BROWSER)
# ============================================================================

class BotCollector:
    """Collects bot URLs from top.gg using Selenium."""
    
    def __init__(self, driver: uc.Chrome, config: ScraperConfig):
        self.driver = driver
        self.config = config
        self.logger = logging.getLogger('topgg_scraper')
    
    def wait_for_page_load(self, url: str):
        """Load page and wait for bot cards to appear."""
        self.driver.get(url)
        try:
            WebDriverWait(self.driver, self.config.page_load_timeout).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/bot/')]"))
            )
        except TimeoutException:
            self.logger.warning("Initial load timeout, checking for Cloudflare...")
            WebDriverWait(self.driver, self.config.cloudflare_timeout).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href,'/bot/')]"))
            )
            self.logger.info("Cloudflare bypass successful")
    
    def extract_bot_urls(self) -> Set[str]:
        """Extract all bot URLs from current page state."""
        urls = set()
        cards = self.driver.find_elements(By.XPATH, "//a[contains(@href,'/bot/')]")
        
        for card in cards:
            try:
                href = card.get_attribute("href")
                cleaned_url = clean_bot_url(href)
                if cleaned_url:
                    urls.add(cleaned_url)
            except Exception as e:
                self.logger.debug(f"Error extracting URL from card: {e}")
        
        return urls
    
    def scroll_and_load_more(self) -> bool:
        """Attempt to load more content via 'Show more' button or scrolling."""
        try:
            show_more = self.driver.find_element(By.XPATH, "//p[contains(text(),'Show more')]")
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", show_more)
            human_sleep(*self.config.scroll_pause)
            show_more.click()
            self.logger.debug("Clicked 'Show more' button")
            return True
        except Exception:
            self.driver.execute_script("window.scrollBy(0, document.body.scrollHeight);")
            return False
    
    def collect_bots_from_page(self, url: str, initial_seen: Set[str], max_bots: int) -> Set[str]:
        """Collect bot URLs from a single page by scrolling."""
        self.logger.info(f"Collecting bots from: {url}")
        self.wait_for_page_load(url)
        
        seen = initial_seen.copy()
        last_count = len(seen)
        stagnant_rounds = 0
        
        with tqdm(desc=f"Collecting from {url.split('/')[-1]}", unit=" bots") as pbar:
            while len(seen) < max_bots:
                new_urls = self.extract_bot_urls()
                seen.update(new_urls)
                
                current_count = len(seen)
                new_found = current_count - last_count
                pbar.update(new_found)
                pbar.set_postfix({'total': current_count, 'stagnant': stagnant_rounds})
                
                if current_count == last_count:
                    stagnant_rounds += 1
                else:
                    stagnant_rounds = 0
                    last_count = current_count
                
                if stagnant_rounds >= self.config.max_stagnant_rounds:
                    self.logger.info(f"No new bots found after {stagnant_rounds} rounds, moving on")
                    break
                
                self.scroll_and_load_more()
                human_sleep(*self.config.scroll_pause)
        
        newly_found = seen - initial_seen
        self.logger.info(f"Found {len(newly_found)} new bots on this page")
        return seen


# ============================================================================
# PARALLEL BOT PAGE DOWNLOADING
# ============================================================================

class ParallelBotDownloader:
    """Downloads bot HTML pages using multiple parallel browsers."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.logger = logging.getLogger('topgg_scraper')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download_single_page(self, driver: uc.Chrome, url: str, worker_id: int) -> Tuple[bool, str]:
        """
        Download a single bot page.
        
        Returns:
            Tuple of (success, bot_id or error_message)
        """
        try:
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extra wait for JS to render
            human_sleep(2, 4)
            
            # Get page source
            html = driver.page_source
            
            # Save to file
            bot_id = url.split('/')[-1]
            filepath = self.output_dir / f"{bot_id}.html"
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html)
            
            return True, bot_id
            
        except Exception as e:
            return False, f"{url}: {str(e)}"
    
    def worker_thread(self, worker_id: int, url_queue: queue.Queue, browser_pool: BrowserPool, 
                     stats: dict, pbar: tqdm):
        """Worker thread that processes URLs using a dedicated browser."""
        driver = browser_pool.get_driver(worker_id)
        if not driver:
            self.logger.error(f"Worker {worker_id}: No driver available")
            return
        
        processed = 0
        
        while True:
            try:
                url = url_queue.get(timeout=1)
            except queue.Empty:
                break
            
            try:
                success, result = self.download_single_page(driver, url, worker_id)
                
                if success:
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
                    stats['failed_urls'].append(result)
                
                processed += 1
                
                # Restart browser periodically
                if processed % self.config.batch_size_per_worker == 0:
                    browser_pool.restart_driver(worker_id)
                    driver = browser_pool.get_driver(worker_id)
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_urls'].append(f"{url}: {str(e)}")
                self.logger.error(f"Worker {worker_id} error: {e}")
            
            finally:
                pbar.update(1)
                pbar.set_postfix({'✓': stats['success'], '✗': stats['failed']})
                url_queue.task_done()
    
    def download_parallel(self, urls: List[str], existing_urls: Set[str], 
                         browser_pool: BrowserPool) -> dict:
        """
        Download multiple bot pages in parallel using browser pool.
        
        Args:
            urls: List of bot URLs to download
            existing_urls: Set of already-downloaded URLs to skip
            browser_pool: Pool of browser instances
            
        Returns:
            Dictionary with download statistics
        """
        urls_to_download = [url for url in urls if url not in existing_urls]
        
        if not urls_to_download:
            self.logger.info("All URLs already downloaded")
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': len(urls)}
        
        self.logger.info(f"Downloading {len(urls_to_download)} pages using {self.config.download_workers} browsers")
        self.logger.info(f"({len(existing_urls)} already exist)")
        
        # Create URL queue
        url_queue = queue.Queue()
        for url in urls_to_download:
            url_queue.put(url)
        
        # Shared statistics
        stats = {
            'total': len(urls_to_download),
            'success': 0,
            'failed': 0,
            'skipped': len(existing_urls),
            'failed_urls': []
        }
        
        # Create progress bar
        with tqdm(total=len(urls_to_download), desc="Downloading bot pages", unit="page") as pbar:
            # Start worker threads
            with ThreadPoolExecutor(max_workers=self.config.download_workers) as executor:
                futures = []
                for worker_id in range(min(self.config.download_workers, len(browser_pool.drivers))):
                    future = executor.submit(
                        self.worker_thread, 
                        worker_id, 
                        url_queue, 
                        browser_pool, 
                        stats, 
                        pbar
                    )
                    futures.append(future)
                
                # Wait for all workers to complete
                for future in futures:
                    future.result()
        
        # Log failed URLs
        if stats['failed_urls']:
            self.logger.warning(f"\n{len(stats['failed_urls'])} downloads failed:")
            for fail in stats['failed_urls'][:10]:
                self.logger.warning(f"  - {fail}")
            if len(stats['failed_urls']) > 10:
                self.logger.warning(f"  ... and {len(stats['failed_urls']) - 10} more")
        
        return stats


# ============================================================================
# MAIN SCRAPER
# ============================================================================

class TopGGScraper:
    """Main scraper orchestrator."""
    
    def __init__(self, config: ScraperConfig, resume: bool = False):
        self.config = config
        self.logger = logging.getLogger('topgg_scraper')
        self.progress_manager = ProgressManager(config)
        self.resume = resume
        
        if resume:
            self.all_bots, self.completed_tags = self.progress_manager.load_progress()
        else:
            self.all_bots = set()
            self.completed_tags = []
        
        existing = self.progress_manager.load_existing_urls()
        self.all_bots.update(existing)
    
    def collect_urls(self, driver: uc.Chrome):
        """Collect bot URLs from top.gg using Selenium."""
        collector = BotCollector(driver, self.config)
        
        # 1. Scrape "New Bots" page
        new_bots_url = "https://top.gg/list/new"
        if "new" not in self.completed_tags:
            self.logger.info("\n" + "="*70)
            self.logger.info("COLLECTING FROM: New Bots Page")
            self.logger.info("="*70)
            
            before = len(self.all_bots)
            self.all_bots = collector.collect_bots_from_page(
                new_bots_url, self.all_bots, self.config.max_bots
            )
            self.completed_tags.append("new")
            self.logger.info(f"New Bots complete: +{len(self.all_bots) - before} bots (total: {len(self.all_bots)})")
            self.progress_manager.save_progress(self.all_bots, self.completed_tags)
        else:
            self.logger.info("Skipping 'New Bots' page (already completed)")
        
        # 2. Scrape tag pages
        remaining_tags = [tag for tag in TAGS if tag not in self.completed_tags]
        self.logger.info(f"\nProcessing {len(remaining_tags)} tag categories...")
        
        for i, tag in enumerate(remaining_tags, 1):
            if len(self.all_bots) >= self.config.max_bots:
                self.logger.info(f"Reached max_bots limit ({self.config.max_bots}), stopping collection")
                break
            
            tag_url = f"https://top.gg/tag/{tag}"
            
            self.logger.info("\n" + "="*70)
            self.logger.info(f"TAG {i}/{len(remaining_tags)}: {tag}")
            self.logger.info("="*70)
            
            before = len(self.all_bots)
            try:
                self.all_bots = collector.collect_bots_from_page(
                    tag_url, self.all_bots, self.config.max_bots
                )
                self.completed_tags.append(tag)
                self.logger.info(f"Tag '{tag}' complete: +{len(self.all_bots) - before} bots (total: {len(self.all_bots)})")
            except Exception as e:
                self.logger.error(f"Failed to process tag '{tag}': {e}")
                continue
            
            self.progress_manager.save_progress(self.all_bots, self.completed_tags)
        
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"URL COLLECTION COMPLETE: {len(self.all_bots)} unique bots found")
        self.logger.info(f"{'='*70}\n")
    
    def download_pages(self, browser_pool: BrowserPool):
        """Download HTML pages using parallel browsers."""
        existing_urls = self.progress_manager.load_existing_urls()
        downloader = ParallelBotDownloader(self.config)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("DOWNLOADING BOT PAGES (PARALLEL)")
        self.logger.info("="*70)
        
        stats = downloader.download_parallel(list(self.all_bots), existing_urls, browser_pool)
        
        self.logger.info("\n" + "="*70)
        self.logger.info("DOWNLOAD SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Total URLs: {len(self.all_bots)}")
        self.logger.info(f"Already existed: {stats['skipped']}")
        self.logger.info(f"Downloaded: {stats['success']}")
        self.logger.info(f"Failed: {stats['failed']}")
        if stats['total'] > 0:
            self.logger.info(f"Success rate: {stats['success']/stats['total']*100:.1f}%")
    
    def run(self):
        """Run the complete scraping process."""
        start_time = time.time()
        browser_pool = None
        collection_driver = None
        
        try:
            self.logger.info("="*70)
            self.logger.info("TOP.GG BOT SCRAPER - ULTIMATE VERSION")
            self.logger.info("="*70)
            self.logger.info(f"Config: {asdict(self.config)}")
            self.logger.info(f"Resume mode: {self.resume}")
            self.logger.info(f"Download workers: {self.config.download_workers} parallel browsers")
            self.logger.info("="*70 + "\n")
            
            # Step 1: Collect URLs (single browser)
            self.logger.info("PHASE 1: URL COLLECTION")
            options = uc.ChromeOptions()
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument("--start-maximized")
            collection_driver = uc.Chrome(options=options)
            collection_driver.set_page_load_timeout(self.config.page_load_timeout)
            
            self.collect_urls(collection_driver)
            
            collection_driver.quit()
            collection_driver = None
            
            # Step 2: Download pages (parallel browsers)
            self.logger.info("\nPHASE 2: PARALLEL DOWNLOADS")
            browser_pool = BrowserPool(self.config.download_workers, self.config)
            browser_pool.initialize()
            
            self.download_pages(browser_pool)
            
            # Final summary
            elapsed = time.time() - start_time
            self.logger.info("\n" + "="*70)
            self.logger.info("SCRAPING COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"Total time: {elapsed/60:.1f} minutes")
            self.logger.info(f"Unique bots: {len(self.all_bots)}")
            self.logger.info(f"HTML files: {len(list(Path(self.config.output_dir).glob('*.html')))}")
            self.logger.info(f"Tags processed: {len(self.completed_tags)}")
            self.logger.info("="*70)
            
        except KeyboardInterrupt:
            self.logger.warning("\n\nScraping interrupted by user")
            self.progress_manager.save_progress(self.all_bots, self.completed_tags)
            self.logger.info("Progress saved. Use --resume to continue later.")
        except Exception as e:
            self.logger.error(f"\n\nFATAL ERROR: {e}", exc_info=True)
            self.progress_manager.save_progress(self.all_bots, self.completed_tags)
            raise
        finally:
            if collection_driver:
                try:
                    collection_driver.quit()
                except:
                    pass
            if browser_pool:
                browser_pool.cleanup()


# ============================================================================
# AUTO-RESTART FUNCTIONALITY
# ============================================================================

def run_with_auto_restart(config: ScraperConfig, resume: bool, interval_hours: float):
    """
    Run scraper with automatic restart every N hours.
    
    Args:
        config: Scraper configuration
        resume: Whether to resume from saved progress
        interval_hours: Hours between restarts (e.g., 1.0 for 1 hour, 0.5 for 30 min)
    """
    logger = logging.getLogger('topgg_scraper')
    cycle_count = 0
    
    while True:
        try:
            cycle_count += 1
            logger.info("\n" + "="*70)
            logger.info(f"🔄 CYCLE #{cycle_count} - Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"⏰ Next restart in {interval_hours} hour(s)")
            logger.info(f"❌ Press Ctrl+C to cancel auto-restart")
            logger.info("="*70 + "\n")
            
            # Run the scraper
            scraper = TopGGScraper(config, resume=resume)
            scraper.run()
            
            # Wait before restart
            logger.info("\n" + "="*70)
            logger.info(f"✅ Cycle #{cycle_count} complete!")
            logger.info(f"💤 Waiting {interval_hours} hour(s) before next cycle...")
            logger.info(f"⏰ Next cycle starts at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + interval_hours * 3600))}")
            logger.info(f"❌ Press Ctrl+C to stop auto-restart")
            logger.info("="*70 + "\n")
            
            time.sleep(interval_hours * 3600)
            
        except KeyboardInterrupt:
            logger.warning("\n" + "="*70)
            logger.warning("⛔ AUTO-RESTART CANCELLED BY USER")
            logger.warning("="*70)
            logger.info(f"Completed {cycle_count} cycle(s)")
            logger.info("Exiting gracefully...")
            break
        except Exception as e:
            logger.error(f"\n❌ ERROR IN CYCLE #{cycle_count}: {e}", exc_info=True)
            logger.info("⏰ Waiting 5 minutes before retry...")
            try:
                time.sleep(300)  # Wait 5 minutes before retry
            except KeyboardInterrupt:
                logger.warning("⛔ Retry cancelled by user. Exiting...")
                break


# ============================================================================
# CLI INTERFACE
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Top.gg Discord Bot Scraper - Ultimate parallel version with AUTO-RESTART',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 3 parallel browsers (recommended)
  python topgg_scraper_ultimate.py --workers 3
  
  # Auto-restart every 1 hour
  python topgg_scraper_ultimate.py --auto-restart 1.0 --workers 3
  
  # Auto-restart every 30 minutes
  python topgg_scraper_ultimate.py --auto-restart 0.5 --workers 3
  
  # Use 5 browsers for maximum speed
  python topgg_scraper_ultimate.py --workers 5 --max-bots 10000
  
  # Resume interrupted scraping with auto-restart
  python topgg_scraper_ultimate.py --resume --auto-restart 1.0
        """
    )
    
    parser.add_argument('--max-bots', type=int, default=45000,
                       help='Maximum number of bots to collect (default: 45000)')
    parser.add_argument('--output-dir', type=str, default='bot_pages',
                       help='Directory to save HTML files (default: bot_pages)')
    parser.add_argument('--workers', type=int, default=3,
                       help='Number of parallel browser instances (default: 3, max: 5)')
    parser.add_argument('--auto-restart', type=float, default=None,
                       help='Auto-restart every N hours (e.g., 1.0 for 1 hour, 0.5 for 30 min)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from saved progress')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, default='scraper.log',
                       help='Log file path (default: scraper.log)')
    
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    
    # Limit workers to reasonable range
    workers = max(1, min(args.workers, 5))
    if workers != args.workers:
        print(f"⚠️  Limiting workers to {workers} (you requested {args.workers})")
    
    config = ScraperConfig(
        max_bots=args.max_bots,
        output_dir=args.output_dir,
        log_file=args.log_file,
        download_workers=workers
    )
    
    logger = setup_logging(config.log_file, args.verbose)
    
    # Run with or without auto-restart
    if args.auto_restart:
        logger.info(f"🔄 AUTO-RESTART ENABLED: Every {args.auto_restart} hour(s)")
        run_with_auto_restart(config, args.resume, args.auto_restart)
    else:
        scraper = TopGGScraper(config, resume=args.resume)
        scraper.run()


if __name__ == "__main__":
    main()