import torch
import tiktoken
import time

from gpt_model import GPTModel
from utils.model import DEFAULT_CFG
from data_loader_v1 import create_dataloader_v1
from pre_train import train_model_simple

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_losses, val_losses, track_tokens_seen = [], [], []

def encode(full_text):
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer.encode(full_text, allowed_special={'<|endoftext|>'})

def init_data_loaders():
    train_file_path = 'train_text_data.txt'
    val_file_path = 'val_text_data.txt'

    with open(train_file_path, "r", encoding="utf-8") as file:
        train_data = file.read()
    with open(val_file_path, "r", encoding="utf-8") as file:
        val_data = file.read()

    train_loader = create_dataloader_v1(
        train_data,
        encode=encode,
        batch_size=12,
        max_length=DEFAULT_CFG["context_length"],
        stride=DEFAULT_CFG["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        encode=encode,
        batch_size=12,
        max_length=DEFAULT_CFG["context_length"],
        stride=DEFAULT_CFG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

def train(train_loader, val_loader,
          num_epochs=10, eval_iter=200, lr=0.0002,
          model_prefix="model_and_optimizer"):

    global train_losses, val_losses, track_tokens_seen

    print(50 * "=")
    print("Starting training...")

    start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(DEFAULT_CFG)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
    lr=3e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.03)

    # Pass train_losses and val_losses as references
    train_model_simple(
        model, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, eval_iter=eval_iter,
        cfg=DEFAULT_CFG, model_prefix=model_prefix,
        train_losses=train_losses, val_losses=val_losses,
        track_tokens_seen=track_tokens_seen,
        warmup_steps=500,                     # shorter warmup
        min_lr_ratio=0.10                     # LR floor at 10% of base
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return model

if __name__ == "__main__":
    train_loader, val_loader = init_data_loaders()
    train(train_loader, val_loader, num_epochs=6,
      eval_iter=10, model_prefix="model_896_14_8_256")