# This implementation follows the omplementation detailed in _Raschka, Sebastian. Build a Large Language Model (From Scratch). Manning Publications, 2024_

#!/usr/bin/env python
# coding: utf-8

import torch
import tiktoken
import os
print("import GPTModel")
from gpt_model import GPTModel
print("import data_loader_v1")  
from data_loader_v1 import create_dataloader_v1
print("import generate")
from generate_text import generate

context_length = 192

# ### Detect if GPU is available

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")


# ### Load training and validation data files

train_file_path = 'train_text_data.txt'
val_file_path = 'val_text_data.txt'

with open(train_file_path, "r", encoding="utf-8") as file:
    train_data = file.read()
with open(val_file_path, "r", encoding="utf-8") as file:
    val_data = file.read()


# ### Initialize data loaders for training
# Data loaders implementation can be found in `./data_loader_v1.py`.
# 

train_ratio = 0.90

# Define the tokenizers
tokenizers = ["tiktoken", "sentencepiece", "bert_base_german"]

# Function to create data loaders
def create_data_loaders(tokenizer_name, train_data, val_data): #, config):
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=4,
        max_length=context_length,
        stride=context_length,
        tokenizer_name=tokenizer_name,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=4,
        max_length=context_length,
        stride=context_length,
        tokenizer_name=tokenizer_name,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


# # Training

from pre_train import train_model_simple
import time

train_losses, val_losses, track_tokens_seen = [], [], []

def train(train_loader, val_loader,
          num_epochs=10, eval_iter=5, 
          sample_text="Every effort moves you",
          checkpoint_path="models/model_and_optimizer.pth", tokenizer="tiktoken"):

    global train_losses, val_losses, track_tokens_seen  # Ensure these are updated globally

    if device == "mps":
        clean()
        print(50 * "=")
        print("Starting training...")

    start_time = time.time()

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)

    # Pass train_losses and val_losses as references
    train_model_simple(
        model, train_loader, val_loader, optimizer,
        num_epochs=num_epochs, eval_iter=eval_iter,
        start_context=sample_text, cfg=GPT_CONFIG_124M,
        checkpoint_path=checkpoint_path,
        train_losses=train_losses, val_losses=val_losses,
        track_tokens_seen=track_tokens_seen, tokenizer=tokenizer

    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    if device == "mps":
        print(50 * "=")
        clean()
    
    return model


# Shuffle through the tokenizers
for tokenizer in tokenizers:

    print(f"Training model with {tokenizer} tokenizer...")
    
    train_loader, val_loader = create_data_loaders(tokenizer, train_data, val_data)#, GPT_CONFIG_124M)

    # ### Set up model configuration 
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": context_length,  # Context length
        "emb_dim": 720, #768,         # Embedding dimension
        "n_heads": 8, #12,          # Number of attention heads
        "n_layers": 8, #12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False,      # Query-Key-Value bias
        "device": device,
    }

    train(train_loader, val_loader, num_epochs=7,
          eval_iter=25, sample_text="Im Park ist",
          checkpoint_path=f"models/model_and_optimizer_{tokenizer}v2.pth", tokenizer=tokenizer)


    # ### Load trained model

    model = GPTModel(GPT_CONFIG_124M)
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.01)

    checkpoint = torch.load(f"models/model_and_optimizer_{tokenizer}v2.pth", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval();

    from generate_text import generate

    torch.set_printoptions(profile="full")
    text = generate(
        model=model, prompt="Er ruft",
        max_new_tokens=50, context_size=GPT_CONFIG_124M['context_length'],
        device="cpu",
        temperature=1,
        top_k=40,
        eos_id=13, 
        tokenizer_name=tokenizer
    )

    splitted = text.split("\n")
    for txt in splitted:
        print(txt)




