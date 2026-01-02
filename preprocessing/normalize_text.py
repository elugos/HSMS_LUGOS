import re
import unicodedata
from typing import List
import spacy


def _basic_normalize(text: str) -> str:
    """
    Cheap pre-normalization before spaCy:
    - Unicode NFKC
    - Lowercase
    - Strip HTML tags
    - Remove URLs/emails
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    # Unicode normalization
    t = unicodedata.normalize("NFKC", text)
    # Lowercase early (helps regex speed)
    t = t.lower()
    # Remove HTML tags
    t = re.sub(r"<[^>]+>", " ", t)
    # Remove URLs/emails
    t = re.sub(r"(https?://\S+|www\.\S+)", " ", t)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", t)
    # Replace non-letter/number with space (keep apostrophes inside words)
    t = re.sub(r"[^a-z0-9'\s]", " ", t)
    # Collapse repeated whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def normalize_text_series(input_texts: List[str], 
                           batch_size: int=50,
                           min_token_len: int=2,
                           n_process: int=8) -> List[List[str]]:
    """
    Normalize a list of texts.
    """

    nlp = spacy.load("en_core_web_lg", disable=["parser", "ner", "textcat"])  # turn off slow modules we won't need
    nlp.enable_pipe("lemmatizer")
    stopwords = nlp.Defaults.stop_words

    _pre = [_basic_normalize(t) for t in input_texts]
    texts = list()
    for doc in nlp.pipe(_pre, batch_size=batch_size, n_process=n_process):
        text = " ".join([token.lemma_ for token in doc if token.is_alpha and token.text not in stopwords and len(token) >= min_token_len])
        texts.append(text)

    return texts

