# %%
import pandas as pd
from pathlib import Path
import yaml

def load_config(path):
    """
    Load event config file and data.

    Returns: Config `cfg` and DataFrame `df`.
    """
    config_file = path
    cfg = yaml.safe_load(Path(config_file).read_text(encoding='utf-8')).get('event', {})
    return cfg


# %%
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
args = parser.parse_args()


config_file = args.input

cfg = load_config(config_file)
cfg

# %%
df = pd.read_csv(cfg['input_file'])

df.columns

# Output, for each 'domain' (url): lists of Actor1Code, Actor1Name, Actor1EthnicCode, <Actor2...>
# EventBaseCode, EventRootCode, GoldsteinScale, AvgTone.




# %%
# Group entries

g = df.groupby('SOURCEURL').agg({
    'Actor1Code': list,
    'Actor1Name': list,
    'Actor1EthnicCode': list,
    'Actor2Code': list,
    'Actor2Name': list,
    'EventCode': list,
    'EventRootCode': list,
    'EventBaseCode': list,
    'GoldsteinScale': list,
    'AvgTone': list,
})
g.head(5)

# %%
config_file = Path(config_file)
output_dir = Path(f"output/{config_file.stem}")

g.reset_index().to_csv(output_dir.joinpath("agg_codes.csv"), index=None)


