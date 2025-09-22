# predict.py
import torch, json
from train_gru import GRULanguageModel, vocab, id2word, PAD_ID, UNK_ID, BOS_ID, EOS_ID, VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, CHECKPOINT_PATH, predict_next_word

# Load checkpoint
ckpt = torch.load("best_gru_checkpoint.pt", map_location="cpu")
model = GRULanguageModel(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, padding_idx=PAD_ID)
model.load_state_dict(ckpt["model_state"])

# sample tausug text
prompt = "aku magkaun"
preds = predict_next_word(model, prompt, top_k=5, device=torch.device("cpu"))
print("Top predictions:", preds)
