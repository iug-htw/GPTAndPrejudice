import torch
import tiktoken
import time

from gpt_model import GPTModel
from data_loader_v1 import create_dataloader_v1
from pre_train import train_model_simple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.2,       # Dropout rate
    "qkv_bias": True,      # Query-Key-Value bias
    "device": DEVICE,
}

train_losses, val_losses, track_tokens_seen = [], [], []

def encode(full_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})

def init_data_loaders():
    train_file_path = 'train_text_data_all_txt.txt'
    val_file_path = 'val_text_data_all_txt.txt'

    with open(train_file_path, "r", encoding="utf-8") as file:
        train_data = file.read()
    with open(val_file_path, "r", encoding="utf-8") as file:
        val_data = file.read()

    train_ratio = 0.90

    train_loader = create_dataloader_v1(
        train_data,
        encode=encode,
        batch_size=4,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        encode=encode,
        batch_size=4,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

def train(train_loader, val_loader,
          num_epochs=10, eval_iter=5, lr=0.0002,
          model_prefix="model_and_optimizer"):

    global train_losses, val_losses, track_tokens_seen

    print(50 * "=")
    print("Starting training...")

    start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-08, weight_decay=0.05)

    # Pass train_losses and val_losses as references
    train_model_simple(
        model, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, eval_iter=eval_iter,
        cfg=GPT_CONFIG_124M, model_prefix=model_prefix,
        train_losses=train_losses, val_losses=val_losses,
        track_tokens_seen=track_tokens_seen,
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return model

if __name__ == "__main__":
    train_loader, val_loader = init_data_loaders()
    train(train_loader, val_loader, num_epochs=30,
      eval_iter=10, model_prefix="gpt_british_lib")