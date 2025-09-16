import sys
import time
import logging
import pandas as pd
from goose3 import Goose
from waybackpy import WaybackMachineCDXServerAPI

# Logging setup
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(message)s',
                    filename='goose_it.log',
                    encoding='utf-8',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Goose instance
g = Goose({
    "browser_user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15)",
    "http_timeout": 15
})

def simplify(domain):
    return domain.replace("www.", "").split(".")[0] if domain else ""

def fetch_from_wayback(url, user_agent="Mozilla/5.0 (Windows NT 5.1; rv:40.0) Gecko/20100101 Firefox/40.0"):
    try:
        cdx = WaybackMachineCDXServerAPI(url, user_agent)
        archive_url = cdx.oldest().archive_url
        logger.info(f"Wayback fallback used for {url}: {archive_url}")
        time.sleep(0.25)
        article = g.extract(url=archive_url)
        return article, archive_url
    except Exception as e:
        logger.error(f"Wayback error for {url}: {e}")
        return None, None

def goose_it(df):
    for i in df.index:
        try:
            url = df.loc[i, 'SOURCEURL']
            article = g.extract(url=url)
        except NetworkError:
            logger.warning(f"NetworkError at index {i}. Trying Wayback for {url}")
            article, archive_url = fetch_from_wayback(url)
            if article is None:
                df.drop(index=i, inplace=True)
                continue
        except Exception as e:
            logger.error(f"Goose error at index {i}: {e}")
            df.drop(index=i, inplace=True)
            continue

        try:
            df.at[i, 'domain'] = article.domain
            df.at[i, 'target'] = simplify(article.domain)
            df.at[i, 'title'] = article.title
            df.at[i, 'text'] = article.cleaned_text
            df.at[i, 'description'] = article.opengraph.get('description', '')
        except Exception as e:
            logger.error(f"Failed to assign fields at index {i}: {e}")
            df.drop(index=i, inplace=True)

    return df

def clean(df, somestr):
    values_to_keep = [193,190,10,12,173,90,112,145,20,40,111,43,42,51,141,11,13,14,15,17,18,19]
    df = df.rename(columns=str.strip)
    df = df[(df['Actor1Geo_CountryCode'] == 'US') | 
            (df['Actor2Geo_CountryCode'] == 'US') | 
            (df['ActionGeo_CountryCode'] == 'US')]
    df = df[df['EventCode'].astype(int).isin(values_to_keep)]
    df = df[df['ActionGeo_FeatureID'] == somestr]
    df = df.drop(columns=[
        'MonthYear', 'FractionDate','Actor1Code', 'Actor1Name','Actor1CountryCode',
        'Actor1KnownGroupCode','Actor1EthnicCode', 'Actor1Religion1Code','Actor1Religion2Code',
        'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code','Actor2Code', 'Actor2Name',
        'Actor2CountryCode', 'Actor2KnownGroupCode','Actor2EthnicCode', 'Actor2Religion1Code',
        'Actor2Religion2Code', 'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code',
        'Actor1Geo_Type', 'Actor1Geo_Fullname','Actor1Geo_ADM1Code', 'Actor1Geo_Lat',
        'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type', 'Actor2Geo_Fullname',
        'Actor2Geo_ADM1Code', 'Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID'
    ], errors='ignore')
    df = df[df['AvgTone'].astype(float) < 1]
    return df


def process_csv(input_csv, output_csv, somestr):
    df = pd.read_csv(input_csv)
    df = clean(df, somestr)
    df = goose_it(df)
    df.to_csv(output_csv, index=False)
    print(f"Output written to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python goose_it.py input.csv output.csv 'feature_id'")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    feature_id = sys.argv[3]

    process_csv(input_csv, output_csv, feature_id)