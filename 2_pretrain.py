#!/usr/bin/env python
# coding: utf-8

# In[111]:


import torch
import tiktoken
import os

from gpt_model import GPTModel
from data_loader_v1 import create_dataloader_v1
from generate_text import generate


# ### Detect if GPU is available

# In[112]:


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")


# ### Set up model configuration 

# In[113]:


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256,  # Context length
    "emb_dim": 384, #768,         # Embedding dimension
    "n_heads": 6, #12,          # Number of attention heads
    "n_layers": 6, #12,         # Number of layers
    "drop_rate": 0.2,       # Dropout rate
    "qkv_bias": False,      # Query-Key-Value bias
    "device": device,
}


# ### Load training and validation data files

# In[114]:


train_file_path = 'train_text_data.txt'
val_file_path = 'val_text_data.txt'

with open(train_file_path, "r", encoding="utf-8") as file:
    train_data = file.read()
with open(val_file_path, "r", encoding="utf-8") as file:
    val_data = file.read()


# ### Initialize data loaders for training
# Data loaders implementation can be found in `./data_loader_v1.py`.
# 
# This implementation follows the omplementation detailed in _Raschka, Sebastian. Build a Large Language Model (From Scratch). Manning Publications, 2024_

# In[115]:


train_ratio = 0.90

train_loader = create_dataloader_v1(
    train_data,
    batch_size=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=4,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)


# ### Initialize the tokenizer

# In[116]:


import tiktoken

#tokenizer = tiktoken.get_encoding("gpt2")
tokenizer = tiktoken.get_encoding("cl100k_base")


total_characters = len(train_data + val_data)
total_tokens = len(tokenizer.encode(train_data + val_data, allowed_special={'<|endoftext|>'}))

print("Characters:", total_characters)
print("Tokens:", total_tokens)


# Print dictionary created by tokenizer

# In[117]:


# Get the token dictionary (token -> token_id mapping)
token_dict = tokenizer._mergeable_ranks

# Print the first few tokens and their IDs
for token, token_id in list(token_dict.items())[:20]:  # Adjust number to see more
    print(f"Token: {repr(token)} -> ID: {token_id}")


# In[118]:


#Decode a Specific Token ID
print(tokenizer.decode([2010]))  


# In[119]:


#View the Full Vocabulary (Sorted by ID)
#sorted_vocab = sorted(token_dict.items(), key=lambda x: x[1])
#for token, token_id in sorted_vocab[:500]:  # Adjust number to see more
#    print(f"ID: {token_id} -> Token: {repr(token)}")


# <h3>Sanity Check</h3>

# In[120]:


# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")


# In[121]:


import gc

def clean(): 
    """
    This is a function for GPU data claening before and after training
    """
    
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    
    gc.collect()  # Force garbage collection
    torch.mps.empty_cache()  # Attempt to release MPS memory
    
    # Move tensors to CPU
    for tensor in list(globals().values()):
        if isinstance(tensor, torch.Tensor) and tensor.device == torch.device("mps"):
            tensor.to("cpu")

    # Delete all tensors
    del tensor
    torch.mps.empty_cache()
    gc.collect()  # Force garbage collection
    print("MPS Available:", torch.backends.mps.is_available())
    print("Allocated Memory:", torch.mps.current_allocated_memory() / (1024**2), "MB")


# # Training

# In[122]:


from pre_train import train_model_simple
import time

train_losses, val_losses, track_tokens_seen = [], [], []

def train(train_loader, val_loader,
          num_epochs=10, eval_iter=5, 
          sample_text="Every effort moves you",
          checkpoint_path="model_and_optimizer.pth"):

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
        track_tokens_seen=track_tokens_seen
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    if device == "mps":
        print(50 * "=")
        clean()
    
    return model


# ### Train the model on training data

# In[ ]:


train(train_loader, val_loader, num_epochs=7,
      eval_iter=25, sample_text="Im Park ist",
      checkpoint_path="model_and_optimizer_9.pth");


# ### Load trained model

# In[ ]:


model = GPTModel(GPT_CONFIG_124M)
model.to("cpu")
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

checkpoint = torch.load("model_and_optimizer_9.pth", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.eval();


# In[ ]:


from generate_text import generate

torch.set_printoptions(profile="full")
text = generate(
    model=model, prompt="Er ruft",
    max_new_tokens=50, context_size=GPT_CONFIG_124M['context_length'],
    device="cpu",
    temperature=1,
    top_k=40,
    eos_id=13
)

splitted = text.split("\n")
for txt in splitted:
    print(txt)


# In[ ]:


if device == "mps":
    clean()


# In[ ]:




