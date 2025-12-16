import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path


from utils import avg_pairwise_distance_by_label
from embedding_models import cosine_distance, train_contrastive_loss, ContrastiveDataset, ContrastiveLoss


import logging
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(message)s',
                    # filename='output.log', 
                    encoding='utf-8', 
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def vectorize(series):
    """
    Takes a pandas Series of embedding_json strings
    and returns a NumPy array of shape (N, 384).
    """
    # Ensure string
    s = series.astype(str).str.strip()

    # Remove brackets, split into float lists
    vecs = (
        s.str[1:-1]                   # strip '[' and ']'
         .str.split(',')              # split by comma
         .apply(lambda x: np.array(x, dtype=np.float32))
         .tolist()
    )

    # Stack into (n_samples, embed_dim)
    return np.vstack(vecs)


def clean_sample(df):
    ## get longest sbert text
    df["text_len"] = df["sbert_text"].str.len()
    df_clean = df.sort_values("text_len", ascending=False).drop_duplicates(subset="GlobalEventID", keep="first")
    df_clean = df_clean.drop(columns=["text_len"])
    ## get all rows that exist in df from global_db and clean to make sure they match, drop all other rows
    # ids = df_clean["GlobalEventID"]
    # global_play_db = global_db[global_db["GlobalEventID"].isin(ids)].copy()
    # valid_ids = set(global_play_db["GlobalEventID"])
    # df_clean = df_clean[df_clean["GlobalEventID"].isin(valid_ids)].copy()
    ## add globaldb to df_clean
    # df_clean = df_clean.merge(
    # global_play_db,
    # on="GlobalEventID",
    # how="left",
    # suffixes=("", "_db")
    # )

    return df_clean



def sample_group(g: pd.DataFrame, df: pd.DataFrame, same: bool=True):
    """
    Generates a sample for a group of EventLabels
    """
    event_label = list(g['EventLabel'].unique())[0]
    if same:
        df_sub = df[df['EventLabel'] == event_label]
    else:
        df_sub = df[df['EventLabel'] != event_label]
    return df_sub.sample(n=len(g), replace=True)


def make_sampler(df: pd.DataFrame, num: int, col: str='embedding_json_title', 
                 balanced_samples: bool=True,
                 balanced_classes: bool=True):
    """
    Generate a set of training samples from a DataFrame.

    This function selects `num` samples from the specified column of the DataFrame,
    optionally balancing the selection across classes or categories.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    num : int
        The total number of samples to generate.
    col : str, optional
        Name of the column containing feature embeddings or text (default: 'embedding_json_title').
    balanced : bool, optional
        If True, attempt to balance positive and negative samples.
    balanced_classes: bool, optional
        If True, attempts to balance classes.
        
    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the sampled rows.
    """
    # df['EventLabel'] = df['EventRootCode'].astype(str).str.split("_").str[-1]

    if not balanced_samples:
        # sample i and j independently
        # This may (albeit unlikely) generate samples where i==j, which is fine, since that would imply the events are the same.
        # Alternatively, we can safely drop those samples, if needed.
        i = df.sample(n=num, random_state=42, replace=True)
        j = df.sample(n=num, random_state=1, replace=True)
    else:
        # Sample a balanced number of pairs
        # Sample positive pairs
        if balanced_classes:
            n_classes = df['EventLabel'].nunique()
            sample_pos = df.groupby('EventLabel').sample(n=num//(2*n_classes), random_state=3, replace=True).reset_index(drop=True)
            sample_neg = df.groupby('EventLabel').sample(n=num//(2*n_classes), random_state=5, replace=True).reset_index(drop=True)
        else:
            sample_pos = df.sample(n=num//2, random_state=7, replace=True).reset_index(drop=True)
            sample_neg = df.sample(n=num//2, random_state=11, replace=True).reset_index(drop=True)

        i_pos = list()
        j_pos = list()
        groups_pos = sample_pos.groupby('EventLabel')
        for name, g in groups_pos:
            i_pos.append(g)
            j_pos.append(sample_group(g, df, True))
        j_pos = pd.concat(j_pos).reset_index()
        i_pos = pd.concat(i_pos).reset_index()

        assert (i_pos['EventLabel']!=j_pos['EventLabel']).sum() == 0
        # sample_neg = df.sample(n=num//2, random_state=13, replace=True)
        i_neg = list()
        j_neg = list()
        groups_neg = sample_neg.groupby('EventLabel')
        for name, g in groups_neg:
            i_neg.append(g)
            j_neg.append(sample_group(g, df, False))
        i_neg = pd.concat(i_neg).reset_index()
        j_neg = pd.concat(j_neg).reset_index()

        assert (i_neg['EventLabel']==j_neg['EventLabel']).sum() == 0

        i = pd.concat([i_pos, i_neg])
        j = pd.concat([j_pos, j_neg])
        
    # reset index so pairs line up
    i = i.reset_index(drop=True)
    j = j.reset_index(drop=True)

    y = (i['EventLabel'].values == j['EventLabel'].values).astype(int)

    # Build sampler
    sampler = pd.DataFrame({
        "GlobalEventID-i": i["GlobalEventID"],
        "GlobalEventID-j": j["GlobalEventID"],
        "features-i": i[col],
        "features-j": j[col],
        "EventCode-i": i['EventLabel'],
        "EventCode-j": j['EventLabel'],
        "y": y
    })

    return sampler



def sample_group(g: pd.DataFrame, df: pd.DataFrame, same: bool=True):
    """
    Generates a sample for a group of EventLabels
    """
    event_label = list(g['EventLabel'].unique())[0]
    if same:
        df_sub = df[df['EventLabel'] == event_label]
    else:
        df_sub = df[df['EventLabel'] != event_label]
    return df_sub.sample(n=len(g), replace=True)


# def make_sampler(df: pd.DataFrame, num: int, col: str='embedding_json_title', balanced: bool=False):
#     """
#     Generate a set of training samples from a DataFrame.

#     This function selects `num` samples from the specified column of the DataFrame,
#     optionally balancing the selection across classes or categories.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         The input DataFrame containing the data.
#     num : int
#         The total number of samples to generate.
#     col : str, optional
#         Name of the column containing feature embeddings or text (default: 'embedding_json_title').
#     balanced : bool, optional
#         If True, attempt to balance samples across classes or categories (default: True).

#     Returns
#     -------
#     pandas.DataFrame
#         A DataFrame containing the sampled rows.
#     """
#     df['EventLabel'] = df['EventRootCode'].astype(str).str.split("_").str[-1]

#     if not balanced:
#         # sample i and j independently
#         # This may (albeit unlikely) generate samples where i==j, which is fine, since that would imply the events are the same.
#         # Alternatively, we can safely drop those samples, if needed.
#         i = df.sample(n=num, random_state=42, replace=True)
#         j = df.sample(n=num, random_state=1, replace=True)
#     else:
#         # Sample a balanced number of pairs
#         # Sample positive pairs
#         sample_pos = df.sample(n=num//2, random_state=0, replace=True)

#         i_pos = list()
#         j_pos = list()
#         groups_pos = sample_pos.groupby('EventLabel')
#         for name, g in groups_pos:
#             i_pos.append(g)
#             j_pos.append(sample_group(g, df, True))
#         j_pos = pd.concat(j_pos).reset_index(drop=True)
#         i_pos = pd.concat(i_pos).reset_index(drop=True)

#         assert (i_pos['EventLabel']!=j_pos['EventLabel']).sum() == 0
#         sample_neg = df.sample(n=num//2, random_state=13, replace=True)
#         i_neg = list()
#         j_neg = list()
#         groups_neg = sample_neg.groupby('EventLabel')
#         for name, g in groups_neg:
#             i_neg.append(g)
#             j_neg.append(sample_group(g, df, False))
#         i_neg = pd.concat(i_neg).reset_index(drop=True)
#         j_neg = pd.concat(j_neg).reset_index(drop=True)

#         assert (i_neg['EventLabel']==j_neg['EventLabel']).sum() == 0

#         i = pd.concat([i_pos, i_neg])
#         j = pd.concat([j_pos, j_neg])
        
#     # reset index so pairs line up
#     i = i.reset_index(drop=True)
#     j = j.reset_index(drop=True)

#     y = (i['EventLabel'].values == j['EventLabel'].values).astype(int)

#     # Build sampler
#     sampler = pd.DataFrame({
#         "GlobalEventID-i": i["GlobalEventID"],
#         "GlobalEventID-j": j["GlobalEventID"],
#         "features-i": i[col],
#         "features-j": j[col],
#         "EventCode-i": i['EventLabel'],
#         "EventCode-j": j['EventLabel'],
#         "y": y
#     })

#     return sampler


