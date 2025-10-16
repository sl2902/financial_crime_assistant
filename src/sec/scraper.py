"""Fetch litigation files from SEC"""

import os
import aiohttp
import asyncio
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
from datetime import datetime, timedelta
import time
from loguru import logger
import re
from typing import Any, List, Dict, Union, Tuple
from dotenv import load_dotenv
load_dotenv()

CONCURRENT_REQUESTS = 5  # Max 5 requests at a time
DELAY_BETWEEN_REQUESTS = 2
ITEMS_PER_PAGE = 100
MAX_PAGES = 5

headers = {
    "User-Agent": 'Mozilla/5.0 (dek.datacamp@email.com)'
}

def scrape_sec_releases(n_page: int) -> List[Dict[str, Any]]:
    """Scrape SEC litigation release links"""
    base_url = "https://www.sec.gov"
    release_url = f"{base_url}/enforcement-litigation/litigation-releases"
    params = {
       "page": n_page,
    }

    releases = []
    try:
        response = requests.get(release_url, headers=headers, params=params)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Error parsing SEC litigation releases {release_url}: {e}")
        raise

    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find('table', attrs={"class": "usa-table views-table views-view-table cols-2"})
    rows = table.find_all('tr')
    releases = []
    for row in rows[1:]:
        cells = row.find_all('td')
        enforcement_url, full_urls, lr_no, lit_date = None, [], None, None
        for cell in cells: 
            respondent_div = cell.select_one("div.release-view__respondents")
            if respondent_div:
                a_tag = respondent_div.find('a', href=True)
                if a_tag:
                    href = a_tag["href"]
                    if 'enforcement-litigation' in href:
                        enforcement_url = base_url + href if href.startswith("/") else href
        
            see_also_div = cell.select_one("div.view-table_subfield_see_also")
            if see_also_div:
                a_tags = see_also_div.find_all('a', href=True)
                for a_tag in a_tags:
                    href = a_tag["href"]
                    if '/files/litigation' in href:
                        full_urls.append(base_url + href if href.startswith("/") else href)
        
            lr_div = cell.select_one("div.view-table_subfield_release_number")
            if lr_div:
                lr_no = lr_div.get_text(strip=True)

            time_tag = cell.find("time")
            if time_tag:
                lit_date = time_tag.get_text(strip=True)
        
        releases.append({
            "date": lit_date,
            "main_link": enforcement_url,
            "lr_no": lr_no,
            "see_also_links": full_urls
        })
    logger.info(f"Fetched SEC litigation links from {n_page * ITEMS_PER_PAGE}-{ITEMS_PER_PAGE * (1 + n_page) - 1}")
    params["start"] = n_page

    return releases

async def scrape_release_content(
      main_link: str, 
      session: aiohttp.ClientSession,
      semaphore: asyncio.Semaphore
    ):
    """Asynchronously scrape content from SEC enforcement HTML page"""
    # limit parallelism
    async with semaphore:
        try:
            async with session.get(main_link, headers=headers) as response:
                if response.status == 403:
                    logger.warning(f"403 Forbidden: {main_link}")
                    return {'title': '', 'content': '', 'success': False}
                    
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                title = soup.find('h1')
                title_text = title.get_text(strip=True) if title else ""

                content = soup.get_text(separator="\n", strip=True)

                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
                return {
                        'title': title_text,
                        'content': content,
                        'success': True
                }
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Error scraping {main_link}: {err_msg}")
            return {'success': False, 'error': err_msg}

def save_releases(releases: List[Dict], filepath: str = "data/raw/sec_releases.json"):
    """Save scraped releases to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {
        "metadata": {
            "scraped_at": datetime.now().isoformat(),
            "total_records": len(releases),
            "source": "SEC Litigation Releases",
        },
        "releases": releases
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(releases)} releases to {filepath}")

async def main():
    logger.info("Started SEC scraping pipeline")
    logger.info("Scrape SEC litigation release metadata")

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    async with aiohttp.ClientSession() as session:
        for idx in range(0, MAX_PAGES):
            tasks = []
            releases = scrape_sec_releases(idx)
            for release in releases:
                main_link = release.get("main_link")
                # logger.info(f"Scraping link {main_link}")
                tasks.append(scrape_release_content(main_link, session, semaphore))
            contents = await asyncio.gather(*tasks)
            msg = 'first' if idx == 0 else 'next'
            logger.info(f"Saving content for the {msg} {ITEMS_PER_PAGE} litigation releases - {idx * ITEMS_PER_PAGE}-{(idx + 1 ) * ITEMS_PER_PAGE - 1}")
            save_releases(contents, filepath=f"data/raw/sec_releases_batch_{idx + 1}.json")

if __name__ == "__main__":
   asyncio.run(main())