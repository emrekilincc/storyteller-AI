# data_prepare_transformer.py
#(1. AŞAMA)
#BU KODLARDA 2 FARKLI DATASET BİRLEŞTİRİLDİ 

import os
import pandas as pd
from datasets import load_dataset


def load_goemotions():
    print("📥 GoEmotions veri seti yükleniyor...")
    goemotions = load_dataset("go_emotions", split="train")
    go_df = pd.DataFrame({
        "text": goemotions["text"],
        "label": [
            goemotions.features["labels"].feature.names[label[0]] if label else "neutral"
            for label in goemotions["labels"]
        ]
    })
    print(f"✅ GoEmotions: {len(go_df)} satır")
    return go_df


def load_dailydialog():
    print("📥 DailyDialog veri seti yükleniyor...")
    dailydialog = load_dataset("daily_dialog", trust_remote_code=True)
    dd_texts, dd_labels = [], []
    emotion_map = {
        0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
        4: "happy", 5: "sad", 6: "surprise"
    }
    for split in ["train", "validation", "test"]:
        for dialog, emotions in zip(dailydialog[split]["dialog"], dailydialog[split]["emotion"]):
            for text, label in zip(dialog, emotions):
                dd_texts.append(text)
                dd_labels.append(emotion_map.get(label, "neutral"))
    dd_df = pd.DataFrame({
        "text": dd_texts,
        "label": dd_labels
    })
    print(f"✅ DailyDialog: {len(dd_df)} satır")
    return dd_df


def clean_text(text):
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9şçöüğğıİŞÖÇÜı: ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_and_clean(go_df, dd_df, output_csv):
    print("📦 Veri setleri birleştiriliyor...")
    merged_df = pd.concat([go_df, dd_df], ignore_index=True)

    # Etiket standardizasyonu
    label_map = {
        "admiration": "happy", "amusement": "happy", "approval": "happy", "caring": "happy",
        "desire": "happy", "excitement": "happy", "gratitude": "happy", "joy": "happy",
        "love": "happy", "optimism": "happy", "pride": "happy", "relief": "happy",
        "surprise": "happy", "happiness": "happy",
        "anger": "anger", "annoyance": "anger", "disapproval": "anger",
        "embarrassment": "sad", "fear": "fear", "grief": "sad", "nervousness": "fear",
        "remorse": "sad", "sadness": "sad", "disgust": "disgust",
        "neutral": "neutral"
    }
    merged_df["label"] = merged_df["label"].str.lower().map(label_map).fillna("neutral")

    # Text'i "duygu: metin" olarak birleştir (prompt conditioning)
    merged_df["text"] = merged_df["label"] + ": " + merged_df["text"].astype(str).str.lower()
    merged_df["text"] = merged_df["text"].apply(clean_text)

    # Son CSV
    merged_df = merged_df[["text"]]
    merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"🎉 Final veri seti oluşturuldu: {output_csv} ({len(merged_df)} satır)")


if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    OUTPUT_CSV = "datasets/merged_transformer_dataset.csv"

    # Veri yükleme
    go_df = load_goemotions()
    dd_df = load_dailydialog()

    # Birleştir ve kaydet
    merge_and_clean(go_df, dd_df, OUTPUT_CSV)
