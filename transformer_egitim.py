# train_transformer.py

import os
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import random
import numpy as np
from model import StoryTransformer

# === Sabit Seed ===
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === PARAMETRELER ===
DATA_PATH = "./datasets/dataset_final.csv"              # BURAYA DATASETİN YOLU YAZILMALI AKSİ HALDE HATA VERİR
TOKENIZER_PATH = "./tokenizer.pickle"
CHECKPOINT_DIR = "M:/DERİN ÖĞRENME PROJE/DERİN ÖĞRENME PROJE/transformer__hikaye/checkpointt"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_transformer_checkpoint.pth")
LOG_PATH = "./loss_log_transformer.csv"
RESUME_TRAINING = True
BATCH_SIZE = 32
EPOCHS = 2
MAX_SEQUENCE_LEN = 60
EMBED_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 6
DROPOUT = 0.1
LEARNING_RATE = 5e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tokenizer'ı yükle ===
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)
vocab_size = len(tokenizer.word_index) + 1

# === Dataset ve temizleme ===
def prepare_sequences(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN, padding='post')
    X = torch.tensor(padded[:, :-1], dtype=torch.long)
    y = torch.tensor(padded[:, 1:], dtype=torch.long)
    return X, y

df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()
split_idx = int(0.9 * len(texts))
train_texts, val_texts = texts[:split_idx], texts[split_idx:]
train_X, train_y = prepare_sequences(train_texts)
val_X, val_y = prepare_sequences(val_texts)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(TextDataset(train_X, train_y), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TextDataset(val_X, val_y), batch_size=BATCH_SIZE)

# === Model ===
print("CUDA kullanılabilir mi?:", torch.cuda.is_available())
print("Kullanılan cihaz:", DEVICE)

model = StoryTransformer(
    vocab_size=vocab_size,
    emotion_size=1,
    embed_size=EMBED_DIM,
    hidden_size=HIDDEN_DIM,
    num_heads=8,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    max_len=MAX_SEQUENCE_LEN
).to(DEVICE)


criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
best_val_loss = float("inf")
epoch_start = 1

# === Metric hesaplama fonksiyonu ===
def calculate_metrics(preds, labels):
    preds = preds.argmax(dim=2).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    acc = (preds == labels).mean() * 100
    f1 = f1_score(labels, preds, average="macro")
    return acc, f1

# === Checkpoint yükleme ===
if RESUME_TRAINING:
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch_") and f.endswith(".pth")]
    if checkpoints:
        latest_ckpt = sorted(checkpoints, key=lambda x: int(x.split('_')[1]))[-1]
        checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_ckpt)
        checkpoint_data = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_data["model_state"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state"])
        best_val_loss = checkpoint_data.get("best_val_loss", float("inf"))
        epoch_start = checkpoint_data["epoch"] + 1
        print(f" Checkpoint yüklendi: {latest_ckpt} (Epoch {epoch_start})")

loss_log = []
early_stopping_patience = 3
epochs_no_improve = 0

print(f" Transformer eğitim başlıyor (veri: {len(df)} satır, {EPOCHS} epoch)...")
for epoch in range(epoch_start, EPOCHS + 1):
    model.train()
    total_loss, total_acc, total_f1 = 0, 0, 0

    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch, torch.zeros(X_batch.size(0), dtype=torch.long).to(DEVICE))
        outputs = outputs[:, 1:, :].reshape(-1, outputs.size(-1))
        targets = y_batch.reshape(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc, f1 = calculate_metrics(outputs.view(X_batch.size(0), -1, vocab_size), y_batch)
        total_acc += acc
        total_f1 += f1

    avg_train_loss = total_loss / len(train_loader)
    avg_train_acc = total_acc / len(train_loader)
    avg_train_f1 = total_f1 / len(train_loader)

    model.eval()
    val_loss, val_acc, val_f1 = 0, 0, 0
    with torch.no_grad():
        for val_Xb, val_yb in val_loader:
            val_Xb, val_yb = val_Xb.to(DEVICE), val_yb.to(DEVICE)
            val_out = model(val_Xb, torch.zeros(val_Xb.size(0), dtype=torch.long).to(DEVICE))
            val_out = val_out[:, 1:, :].reshape(-1, val_out.size(-1))
            targets = val_yb.reshape(-1)
            vloss = criterion(val_out, targets)
            val_loss += vloss.item()
            acc, f1 = calculate_metrics(val_out.view(val_Xb.size(0), -1, vocab_size), val_yb)
            val_acc += acc
            val_f1 += f1

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    avg_val_f1 = val_f1 / len(val_loader)

    print(f"\n Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
          f"Train Acc: {avg_train_acc:.2f}% | Val Acc: {avg_val_acc:.2f}% | "
          f"Train F1: {avg_train_f1:.4f} | Val F1: {avg_val_f1:.4f}")

    loss_log.append({
        "epoch": epoch,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "train_acc": avg_train_acc,
        "val_acc": avg_val_acc,
        "train_f1": avg_train_f1,
        "val_f1": avg_val_f1
    })

    # === Checkpoint kaydı ===
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss
    }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch}_checkpoint.pth"))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(" En iyi model checkpoint olarak kaydedildi.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if avg_train_loss < avg_val_loss:
            print(" Overfitting belirtisi: Val loss > Train loss.")

        if epochs_no_improve >= early_stopping_patience:
            print(" Early stopping: Overfitting tespit edildi. Eğitim durduruluyor.")
            break

pd.DataFrame(loss_log).to_csv(LOG_PATH, index=False)
print(" Kayıtlar CSV'ye yazıldı:", LOG_PATH)

# === Grafikler ===
plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_loss"] for l in loss_log], label="Train Loss")
plt.plot([l["epoch"] for l in loss_log], [l["val_loss"] for l in loss_log], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Grafiği (Transformer)")
plt.savefig("loss_plot_transformer.png")
plt.close()

plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_acc"] for l in loss_log], label="Train Acc (%)")
plt.plot([l["epoch"] for l in loss_log], [l["val_acc"] for l in loss_log], label="Val Acc (%)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Grafiği (Transformer)")
plt.savefig("accuracy_plot_transformer.png")
plt.close()

plt.figure()
plt.plot([l["epoch"] for l in loss_log], [l["train_f1"] for l in loss_log], label="Train F1")
plt.plot([l["epoch"] for l in loss_log], [l["val_f1"] for l in loss_log], label="Val F1")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.legend()
plt.title("F1-Score Grafiği (Transformer)")
plt.savefig("f1_plot_transformer.png")
plt.close()

print(" Transformer eğitimi tamamlandı ve grafikler oluşturuldu.")
