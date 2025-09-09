import torch
from gpt_model import GPTModel

# YOU UPDATE THIS WITH THE LATEST MODEL CONFIGUATIONS 
# ===================================================
DEFAULT_CFG = {
  "vocab_size": 50257,
  "context_length": 256,
  "emb_dim": 896,
  "n_heads": 14,
  "n_layers": 8,
  "drop_rate": 0.2,
  "qkv_bias": True
}
# ===================================================

def load_GPT_model(path, cfg=DEFAULT_CFG, device="cpu", eval=True):
    model = GPTModel(cfg)
    checkpoint = torch.load(path, weights_only=True, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if eval:
        model.eval()

    return model