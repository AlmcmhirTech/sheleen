# dataloader_test.py
# Small, well-commented test to show tokenization, dataset, padding, and DataLoader.

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ----- CONFIG: file names (change if your filenames differ) -----
VOCAB_FILE = "vocab.json"
TRAIN_FILE = "train.txt"
BATCH_SIZE = 4   # small for testing; later you can increase

# ----- Load vocab -----
with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# reverse mapping (id -> token) useful for debugging
id2word = {idx: word for word, idx in vocab.items()}

PAD_ID  = vocab.get("<PAD>", 0)
UNK_ID  = vocab.get("<UNK>", 1)
BOS_ID  = vocab.get("<BOS>", 2)
EOS_ID  = vocab.get("<EOS>", 3)

# ----- Simple tokenizer function -----
def tokenize_sentence(sentence):
    """
    Lowercases and splits on whitespace. Returns list of tokens with BOS/EOS.
    (Assumes sentences in train.txt are already normalized.)
    """
    sentence = sentence.strip().lower()
    if sentence == "":
        return []
    tokens = ["<BOS>"] + sentence.split() + ["<EOS>"]
    return tokens

# ----- Convert tokens to ids -----
def tokens_to_ids(tokens):
    return [vocab.get(t, UNK_ID) for t in tokens]

# ----- Custom Dataset -----
class TausugDataset(Dataset):
    def __init__(self, txt_path):
        # load sentences (one per line)
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        self.sentences = lines

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        tokens = tokenize_sentence(sent)          # ["<BOS>", "aku", ... , "<EOS>"]
        ids = tokens_to_ids(tokens)               # [2, 4887, ... , 3]
        return torch.tensor(ids, dtype=torch.long)

# ----- Collate function for padding a batch -----
def collate_fn(batch):
    """
    batch: list of 1D tensors with different lengths
    returns:
      padded (batch_size, max_len)
      mask   (batch_size, max_len) where 1 = real token, 0 = pad
    """
    # pad_sequence pads to the max length in batch
    padded = pad_sequence(batch, batch_first=True, padding_value=PAD_ID)
    mask = (padded != PAD_ID).long()
    return padded, mask

# ----- Run a quick test -----
def main():
    dataset = TausugDataset(TRAIN_FILE)
    print(f"[INFO] Loaded dataset with {len(dataset)} sentences.")

    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            collate_fn=collate_fn)

    # get one batch and print debug info
    for batch_idx, (padded_batch, mask) in enumerate(dataloader):
        print(f"\n[DEBUG] Batch {batch_idx} (showing first batch only)")
        print("Padded batch shape:", padded_batch.shape)  # (batch_size, max_len)
        print("Mask shape:", mask.shape)
        print("\nPadded batch (IDs):")
        print(padded_batch)
        print("\nMask (1=token,0=pad):")
        print(mask)

        # show decoded words for the first sentence in the batch
        first = padded_batch[0].tolist()
        print("\nDecoded first sequence (tokens):")
        print([id2word.get(i, "<UNK>") for i in first])

        # stop after the first batch (this is a test)
        break

if __name__ == "__main__":
    main()