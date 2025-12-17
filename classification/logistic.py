
# file: baseline_tfidf_logreg.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np


LABELS = [
    "CONSULT","ASSAULT","DISAPPROVE","SUPPORT","COERCE","AGREE","AID","REJECT",
    "CONCEDE","COOPERATE","RETREAT","THREATHEN","DEMAND","PROTEST","SANCTION","MOBILIZE"
]


def prepare_dataset(df: pd.DataFrame):
    # Feature: title (or title + first_para concatenated)
    # Targets: one-hot over 16 classes
    df["EventLabel"] = df["EventLabel"].astype(str).str.upper()


    # Text feature: title + first paragraph
    df["text_feat"] = df["title"].fillna("") + " " + df.get("first_para","").fillna("")

    # Aggregate articles of the same label
    key = "SOURCEURL"
    agg = (
        df.groupby(key)
        .agg({
            "text_feat":"first",  # or join if multiple variants exist
            "EventLabel": lambda s: list(set(s))  # unique labels per article
        })
        .reset_index()
    )

    mlb = MultiLabelBinarizer(classes=LABELS)
    Y = mlb.fit_transform(agg['EventLabel'])
    X_text = agg['text_feat'].fillna("").astype(str)

    return X_text, Y


def train_tfidf_logistic(X_text, Y_tr):

    # X_tr, X_va, Y_tr, Y_va = train_test_split(text, Y, test_size=0.2, random_state=42, stratify=y_raw)

    clf = make_pipeline(
        TfidfVectorizer(max_features=1000, ngram_range=(1,2), lowercase=True),
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=2000, C=2.0,
                class_weight="balanced", solver="liblinear"
            )
        )
    )

    # tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2), lowercase=True)
    # Xtr = tfidf.fit_transform(X_text)
    # # Xva = tfidf.transform(X_va)

    # # Class-weighted logistic heads
    # clf = OneVsRestClassifier(
    #     LogisticRegression(
    #         max_iter=2000, C=2.0, class_weight="balanced", solver="liblinear"
    #     )
    # )
    clf.fit(X_text, Y_tr)
    # Yp = clf.predict(Xva)
    return clf




def tune_thresholds_f1(Y_true, Y_proba, grid=None):
    """
    Y_true: ndarray [N, C] with {0,1}
    Y_proba: ndarray [N, C] with probabilities [0,1]
    grid: iterable of thresholds; default 0.05..0.95
    Returns: ndarray [C] of best thresholds that maximize F1 per class.
    """
    if grid is None:
        grid = np.linspace(0.05, 0.95, 21)
    N, C = Y_true.shape
    best = np.zeros(C, dtype=np.float32)
    for c in range(C):
        y = Y_true[:, c].astype(np.int32)
        p = Y_proba[:, c]
        best_f1, best_t = -1.0, 0.5
        # Pre-sort for faster PR sweep (optional)
        for t in grid:
            pred = (p >= t).astype(np.int32)
            tp = (pred & y).sum()
            fp = (pred & (1 - y)).sum()
            fn = ((1 - pred) & y).sum()
            precision = tp / max(tp + fp, 1e-9)
            recall    = tp / max(tp + fn, 1e-9)
            f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best[c] =        best[c] = best_t
    
    return best
