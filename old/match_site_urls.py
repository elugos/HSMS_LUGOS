import re
import csv
from urllib.parse import urlparse
import pandas as pd


"""
This script uses files `mbfc_source_data.csv` and `extra_news.txt`
and attempts to match sources in both lists.

"""
import logging

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s|%(process)d|%(name)s|%(levelname)s|%(message)s',
 #                   filename='mbfc_search_scraper.log', 
                    encoding='utf-8', 
                    level=logging.DEBUG)

if __name__ == "__main__":

    extra_news = list()

    with open('data/extra_news.txt', "r") as fin:
    
        re_clean = re.compile(r"(www\.)")  # regex for cleaning up URLs

        for line in fin:  # read sources line by line
            extra_news.append(line.strip())  # Parse URLs in each line
    logger.info(f"Read {len(extra_news)} extra news sites: {extra_news[:3]} ...")
        
    # Read source data CSV into a Pandas DataFrame
    df = pd.read_csv("data/mbfc_source_data.csv")
    df = df.dropna(subset=['website'])  # drop rows where website is invalid
    
    for u in df['website']:
        print(u)
        print(urlparse(u))
    
    
