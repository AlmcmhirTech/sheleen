import json
import random
import os

# ----------------------------
# CONFIG
# ----------------------------
CORPUS_FILE = "sample_data.txt"
VOCAB_FILE = "vocab.json"
TRAIN_FILE = "train.txt"
VAL_FILE = "val.txt"
TEST_FILE = "test.txt"

# reproducibility
random.seed(42)

# ----------------------------
# LOAD CORPUS
# ----------------------------
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"[INFO] Loaded {len(sentences)} sentences from {CORPUS_FILE}")

# ----------------------------
# BUILD VOCABULARY
# ----------------------------
word_freq = {}
for sent in sentences:
    for word in sent.split():
        word_freq[word] = word_freq.get(word, 0) + 1

# assign IDs
# reserve 0 for <PAD>, 1 for <UNK>, 2 for <BOS>, 3 for <EOS>
vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
for i, word in enumerate(sorted(word_freq.keys()), start=4):
    vocab[word] = i

# save vocab
with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"[INFO] Vocabulary size: {len(vocab)} (including special tokens)")

# ----------------------------
# SPLIT DATA (train/val/test)
# ----------------------------
random.shuffle(sentences)

n_total = len(sentences)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

train_data = sentences[:n_train]
val_data = sentences[n_train:n_train+n_val]
test_data = sentences[n_train+n_val:]

def save_list(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        for s in data:
            f.write(s + "\n")

save_list(TRAIN_FILE, train_data)
save_list(VAL_FILE, val_data)
save_list(TEST_FILE, test_data)

print(f"[INFO] Train/Val/Test sizes: {len(train_data)} / {len(val_data)} / {len(test_data)}")
