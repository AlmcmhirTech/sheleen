# train_gru.py
"""
Train a simple GRU-based next-word predictor on your Tausug dataset.

How it works (high-level):
- Load vocab.json and train.txt (one sentence per line, already tokenized/normalized).
- Dataset returns ID tensors like: [<BOS>, w1, w2, ..., <EOS>]
- Collate pads the batch to (batch_size, max_len)
- For training we use inputs = batch[:, :-1], targets = batch[:, 1:]
- Loss: CrossEntropyLoss(ignore_index=PAD_ID)
- Checkpoint the best model by validation loss.
"""

import os
import json
import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_

# ----------------------------
# CONFIG (edit if needed)
# ----------------------------
VOCAB_FILE = "vocab.json"
TRAIN_FILE = "train.txt"
VAL_FILE = "val.txt"
CHECKPOINT_PATH = "best_gru_checkpoint.pt"

# hyperparameters (start conservative for CPU)
EMBED_SIZE = 128
HIDDEN_SIZE = 128
NUM_LAYERS = 1
BATCH_SIZE = 16      # reduce to 8/16 if CPU too slow
LR = 1e-3
EPOCHS = 10
GRAD_CLIP = 1.0
SEED = 42

# reproducibility
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------
# Helper: load vocab & mappings
# ----------------------------
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# word -> id (vocab), create id -> word mapping (for sampling)
id2word = {v: k for k, v in vocab.items()}
PAD_ID = vocab.get("<PAD>", 0)
UNK_ID = vocab.get("<UNK>", 1)
BOS_ID = vocab.get("<BOS>", 2)
EOS_ID = vocab.get("<EOS>", 3)
VOCAB_SIZE = len(vocab)

# ----------------------------
# Simple Dataset (same as your test)
# ----------------------------
class TausugDataset(Dataset):
    def __init__(self, txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.sentences = lines

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx].lower().strip()
        tokens = ["<BOS>"] + sent.split() + ["<EOS>"]
        ids = [vocab.get(t, UNK_ID) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    """
    Pads a list of 1D tensors into a (batch_size, max_len) tensor.
    Returns: padded tensor, mask (1 for real token, 0 for PAD)
    """
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    mask = (padded != PAD_ID).long()
    return padded, mask

# ----------------------------
# GRU Model
# ----------------------------
class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, padding_idx=0, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        """
        x: [batch_size, seq_len] (IDs)
        returns:
          logits: [batch_size, seq_len, vocab_size]
          hidden: GRU hidden states
        """
        emb = self.embedding(x)                    # [B, S, E]
        out, hidden = self.gru(emb, hidden)        # out: [B, S, H]
        out = self.dropout(out)
        logits = self.fc(out)                      # [B, S, V]
        return logits, hidden

# ----------------------------
# Training & Evaluation helpers
# ----------------------------
def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for padded, mask in dataloader:
            padded = padded.to(device)
            # inputs -> all except last token; targets -> all except first token
            inputs = padded[:, :-1]
            targets = padded[:, 1:]
            logits, _ = model(inputs)
            # reshape to (N*T, V) and (N*T,)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            total_loss += loss.item()
            n_batches += 1
    avg_loss = total_loss / max(1, n_batches)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, perplexity

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # create datasets & loaders
    train_ds = TausugDataset(TRAIN_FILE)
    val_ds = TausugDataset(VAL_FILE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Train size: {len(train_ds)}  Val size: {len(val_ds)}  Vocab size: {VOCAB_SIZE}")

    model = GRULanguageModel(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, padding_idx=PAD_ID)
    model = model.to(device)

    # ignore pad when computing cross-entropy
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")
        for padded, mask in pbar:
            # padded: [B, S]
            padded = padded.to(device)
            inputs = padded[:, :-1]   # input tokens
            targets = padded[:, 1:]   # next-token targets

            logits, _ = model(inputs) # logits: [B, S-1, V]
            # compute loss
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping to stabilize training
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(train_loss = f"{running_loss / ( (pbar.n+1) ):0.4f}")

        avg_train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_ppl = evaluate(model, val_loader, device, criterion)

        print(f"\nEpoch {epoch} summary: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}")

        # checkpoint if improved
        if val_loss < best_val_loss:
            print("Validation loss improved -> saving checkpoint.")
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab": vocab,
                "config": {
                    "embed_size": EMBED_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS
                }
            }, CHECKPOINT_PATH)
        else:
            print("No validation improvement this epoch.")

    print("Training finished.")

# ----------------------------
# Simple inference: predict next word
# ----------------------------
def predict_next_word(model, prompt, top_k=5, device=None):
    """
    Given a prompt string (normalized/lowercase), return top_k predicted next words with probabilities.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    tokens = ["<BOS>"] + prompt.strip().split()
    ids = [vocab.get(t, UNK_ID) for t in tokens]
    tensor_in = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        logits, _ = model(tensor_in)  # [1, seq_len, V]
        last_logits = logits[0, -1, :]  # last time step
        probs = torch.softmax(last_logits, dim=-1)
        topk = torch.topk(probs, k=top_k)
        top_ids = topk.indices.tolist()
        top_probs = topk.values.tolist()

    return [(id2word.get(i, "<UNK>"), p) for i, p in zip(top_ids, top_probs)]

# ----------------------------
# Run training when invoked
# ----------------------------
if __name__ == "__main__":
    train()
