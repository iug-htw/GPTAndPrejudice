import torch
import tiktoken

def _enc():
    return tiktoken.get_encoding("gpt2")

def text_to_token_ids(text, tokenizer=None):
    if tokenizer == None:
        tokenizer = _enc()

    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer=None):
    if tokenizer == None:
        tokenizer = _enc()

    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())