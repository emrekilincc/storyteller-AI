# tokenizer.py
# (2. AŞAMA)
# BU KODLARDA VERİLER TEMİZLENDİ, TOKENIZER OLUŞTURULDU, VOCAB KAYDEDİLDİ

import pandas as pd
from collections import Counter
import json
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

def build_vocab(texts, min_freq=2):
    word_count = Counter()
    for text in texts:
        tokens = text.strip().split()
        word_count.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    idx = 4
    for word, count in word_count.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"✅ Vocab kaydedildi: {path} (toplam {len(vocab)} kelime)")

def main():
    # Merged veri setini yükle
    data_df = pd.read_csv("datasets/merged_transformer_dataset.csv", encoding="utf-8-sig")
    texts = data_df["text"].tolist()

    # Vocab oluştur ve kaydet
    vocab = build_vocab(texts, min_freq=2)
    save_vocab(vocab, "vocab.json")

    # Tokenizer oluştur ve kaydet
    tokenizer = Tokenizer(oov_token="<unk>")
    tokenizer.fit_on_texts(texts)
    with open("tokenizer.pickle", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer kaydedildi: tokenizer.pickle")

if __name__ == "__main__":
    main()
