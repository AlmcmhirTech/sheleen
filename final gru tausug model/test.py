import json
import torch

# Load vocab
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Reverse mapping (ID → word) for debugging
id2word = {idx: word for word, idx in vocab.items()}

# Example sentence
sentence = "hambuuk sin magtutud"

# Add BOS and EOS
tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]
print("Tokens:", tokens)

# Convert to IDs
ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
print("IDs:", ids)

# Convert to tensor
tensor = torch.tensor(ids, dtype=torch.long)
print("Tensor:", tensor)

# Optional: convert back for sanity check
decoded = [id2word[i] for i in ids]
print("Decoded back:", decoded)
