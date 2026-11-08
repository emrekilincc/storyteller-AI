"""Check the archived Transformer weights without training or downloading a model."""
from pathlib import Path
import pickle

import torch
from model import StoryTransformer


def main():
    root = Path(__file__).resolve().parent
    with (root / "tokenizer.pickle").open("rb") as file:
        tokenizer = pickle.load(file)
    state = torch.load(
        root / "best_transformer_checkpoint.pth",
        map_location="cpu",
        weights_only=True,
    )
    model = StoryTransformer(
        vocab_size=len(tokenizer.word_index) + 1,
        emotion_size=1,
        embed_size=256,
        hidden_size=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        max_len=60,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    tokens = tokenizer.texts_to_sequences(["happy: once upon a time"])[0]
    with torch.no_grad():
        output = model(torch.tensor([tokens]), torch.zeros(1, dtype=torch.long))
    if output.shape != (1, len(tokens) + 1, len(tokenizer.word_index) + 1):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains non-finite values.")
    print("OK: checkpoint and tokenizer match; forward pass is finite.")


if __name__ == "__main__":
    main()
