"""
Loads data from https://pmc.ncbi.nlm.nih.gov/articles/PMC11402535/
These are news articles posted on Reddit.
"""

import pandas as pd


path ="data/reddit/Raw and Labeled Data/Liberal.json"

df = pd.read_json(path)

print(df.head())