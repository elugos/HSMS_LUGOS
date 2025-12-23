import sys
import argparse
import os
import time
import logging
import pandas as pd
from goose3 import Goose
from goose3.network import NetworkError
from waybackpy import WaybackMachineCDXServerAPI

map = {
    0: "GlobalEventID", 
    1: "date", 
    2: "MonthYear", 
    3: "Year", 
    4: "FractionDate", 
    5: "Actor1Code", 
    6: "Actor1Name", 
    7: "Actor1CountryCode", 
    8: "Actor1KnownGroupCode", 
    9: "Actor1EthnicCode",
    10: "Actor1Religion1Code", 
    11: "Actor1Religion2Code", 
    12: "Actor1Type1Code", 
    13: "Actor1Type2Code",
    14: "Actor1Type3Code", 
    15: "Actor2Code", 
    16: "Actor2Name", 
    17: "Actor2CountryCode", 
    18: "Actor2KnownGroupCode", 
    19: "Actor2EthnicCode",
    20: "Actor2Religion1Code", 
    21: "Actor2Religion2Code", 
    22: "Actor2Type1Code", 
    23: "Actor2Type2Code", 
    24: "Actor2Type3Code", 
    25: "IsRootEvent", 
    26: "EventCode", 
    27: "EventBaseCode", 
    28: "EventRootCode", 
    29: "QuadClass",
    30: "GoldsteinScale", 
    31: "NumMentions", 
    32: "NumSources", 
    33: "NumArticles", 
    34: "AvgTone", 
    35: "Actor1Geo_Type", 
    36: "Actor1Geo_Fullname", 
    37: "Actor1Geo_CountryCode", 
    38: "Actor1Geo_ADM1Code", 
    39: "Actor1Geo_Lat",
    40: "Actor1Geo_Long", 
    41: "Actor1Geo_FeatureID", 
    42: "Actor2Geo_Type", 
    43: "Actor2Geo_Fullname", 
    44: "Actor2Geo_CountryCode", 
    45: "Actor2Geo_ADM1Code", 
    46: "Actor2Geo_Lat",
    47: "Actor2Geo_Long",
    48: "Actor2Geo_FeatureID", 
    49: "ActionGeo_Type", 
    50: "ActionGeo_Fullname", 
    51: "ActionGeo_CountryCode", 
    52: "ActionGeo_ADM1Code", 
    53: "ActionGeo_Lat", 
    54: "ActionGeo_Long", 
    55: "ActionGeo_FeatureID",
    56: "DATEADDED", 
    57: "SOURCEURL"
}

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
            df.at[i, 'datetime'] = article.publish_date
        except Exception as e:
            logger.error(f"Failed to assign fields at index {i}: {e}")
            df.drop(index=i, inplace=True)

    return df

def clean(df, somestr):
    df = df.rename(columns=map)
    df = df[(df['Actor1Geo_CountryCode'] == 'US') | 
            (df['Actor2Geo_CountryCode'] == 'US') | 
            (df['ActionGeo_CountryCode'] == 'US')]
    df = df[df['ActionGeo_FeatureID'] == somestr]
    df = df.drop(columns=[
        'MonthYear', 'FractionDate',
        'Actor1KnownGroupCode', 'Actor1Religion1Code','Actor1Religion2Code',
        'Actor1Type1Code','Actor1Type2Code','Actor1Type3Code',
        'Actor2KnownGroupCode', 'Actor2Religion1Code',
        'Actor2Religion2Code', 'Actor2Type1Code','Actor2Type2Code','Actor2Type3Code',
        'Actor1Geo_Type', 'Actor1Geo_Fullname','Actor1Geo_ADM1Code', 'Actor1Geo_Lat',
        'Actor1Geo_Long', 'Actor1Geo_FeatureID', 'Actor2Geo_Type', 'Actor2Geo_Fullname',
        'Actor2Geo_ADM1Code', 'Actor2Geo_Lat','Actor2Geo_Long','Actor2Geo_FeatureID'
    ], errors='ignore')
    return df


def process_csv(input_csv, output_csv, somestr):
    df = pd.read_csv(input_csv,sep='\t',header=None)
    df = df.rename(columns=map)

    df = clean(df, somestr)
    df = goose_it(df)
    df.to_csv(output_csv, index=False)
    logger.info(f"Output written to {output_csv}")


def process_one(file, input_dir, output_dir, feature_id):
    """Process one CSV file and write to output directory."""
    input_path = os.path.join(input_dir, file)
    date_part = os.path.splitext(file)[0]
    output_path = os.path.join(output_dir, f"{date_part}_goosed.csv")

    logger.info(f"📄 Processing {file}...")

    try:
        df = pd.read_csv(input_path,sep='\t',header=None)
        df = clean(df, feature_id)
        df = goose_it(df)
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Finished {file} → {output_path}")
    except Exception as e:
        logger.info(f"⚠️ Failed on {file}: {e}")

def process_all(input_dir, output_dir, feature_id):
    """Sequentially process all CSVs in the input directory."""
    os.makedirs(output_dir, exist_ok=True)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith(".CSV")]

    if not csv_files:
        logger.info(f"❌ No CSV files found in {input_dir}")
        return

    logger.info(f"📂 Found {len(csv_files)} CSV files in {input_dir}")
    logger.info(f"🌎 Filtering for feature_id = '{feature_id}'\n")

    for file in csv_files:
        # Get output_path 
        date_part = os.path.splitext(file)[0]
        output_path = os.path.join(output_dir, f"{date_part}_goosed.csv")
        if os.path.exists(output_path):  # Skip if output_file already exists! Do not rescrape it.
            logger.info(f"Skipping file {file} - output {output_path} already exists!")
            continue
        process_one(file, input_dir, output_dir, feature_id)
        time.sleep(0.5)
    logger.info("\n🎉 All files processed. Results saved to:", output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Batch scrape GDELT CSVs with Goose3 + Wayback fallback (sequential version)."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input directory containing CSV files"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for processed CSVs"
    )
    parser.add_argument(
        "--feature", "-f", required=True,
        help="ActionGeo_FeatureID to filter on (e.g., 'VA', 'USNYC')"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)

    process_all(input_dir, output_dir, args.feature)


if __name__ == "__main__":
    main()
