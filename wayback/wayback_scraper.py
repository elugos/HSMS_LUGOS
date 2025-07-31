import json
import time

from waybackpy import WaybackMachineCDXServerAPI
from goose3 import Goose

import logging

# Set up logger
logging.basicConfig(format='%(asctime)s|%(process)d|%(name)s|%(levelname)s|%(message)s',
                    filename='wayback_scraper.log', 
                    encoding='utf-8', 
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    user_agent = "Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"

    with open('urls.txt') as fin:
        urls = list(map(lambda s: s.strip(), fin.readlines()))

    fout = open("wayback_articles.jsonl", "w")

    g = Goose({
            "browser_user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:131.0) Gecko/20100101 Firefox/131.0",  # Goose user-agent
            "http_timeout": 15  # Timeout in seconds
            })

    for i, url in enumerate(urls):
        cdx = WaybackMachineCDXServerAPI(url, user_agent)

        archive_url = cdx.oldest().archive_url
        
        logger.info(f"Found archive url: {archive_url}. Attempting to scrape...")
        
        try:
            time.sleep(0.25)  # Sleep
            gooseArticle = g.extract(url=archive_url)
            logger.debug(f"Got article {gooseArticle.title}")

            a = dict()
            a['url'] = url
            a['archive_url'] = archive_url
            a['title'] = gooseArticle.title
            a['content'] = gooseArticle.cleaned_text
            a['date'] = gooseArticle.publish_date

            fout.write(f"{json.dumps(a)}\n")
        except ValueError as e:
            logger.error(f"Invalid URL: {archive_url}: {e}") 
        except Exception as e:
            logger.error(f"Error while scraping {archive_url}: {e}")

    fout.close()
