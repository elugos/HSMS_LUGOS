# hatebert_classify.py
import sys
import torch
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_texts_and_save_from_file(input_filename, output_filename):
    df = pd.read_csv(input_filename)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
    model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")

    preds, ratios = [], []

    for text in df["text"]:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            preds.append([])
            ratios.append({})
            continue

        inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(logits, dim=-1).tolist()

        preds.append(pred)
        count = Counter(pred)
        ratios.append({k: round(v / len(pred), 3) for k, v in count.items()})

    df["predicted_class"] = preds
    df["class_ratio"] = ratios
    df.to_csv(output_filename, index=False)
    print(f"Saved to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hatebert_classify.py input.csv output.csv")
        sys.exit(1)
    classify_texts_and_save_from_file(sys.argv[1], sys.argv[2])
