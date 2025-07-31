import sys
import torch
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def classify_texts_and_save_from_file(input_filename, output_filename):
    # Load CSV
    df = pd.read_csv(input_filename)

    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

    predictions_per_row = []
    class_ratios_per_row = []

    for full_text in df["text"]:
        # Split text into non-empty lines
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        
        if not lines:
            predictions_per_row.append([])
            class_ratios_per_row.append({})
            continue

        # Tokenize lines
        inputs = tokenizer(lines, padding=True, truncation=True, return_tensors="pt")

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = torch.argmax(probs, dim=-1)

        predictions = predicted_classes.tolist()
        predictions_per_row.append(predictions)

        # Calculate class ratio
        total = len(predictions)
        counter = Counter(predictions)
        ratio = {cls: round(count / total, 3) for cls, count in counter.items()}
        class_ratios_per_row.append(ratio)

    # Append new columns to DataFrame
    df["predicted_class"] = predictions_per_row
    df["class_ratio"] = class_ratios_per_row

    # Save to output
    df.to_csv(output_filename, index=False)
    print(f"Saved predictions and ratios to {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transformer.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    classify_texts_and_save_from_file(input_file, output_file)