import tiktoken
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer  # For Hugging Face tokenizers

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, tokenizer_type="tiktoken"):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the text based on tokenizer type
        if tokenizer_type == "tiktoken":
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        elif tokenizer_type == "sentencepiece":
            token_ids = tokenizer.encode(txt)
        elif tokenizer_type == "bert_base_german":
            token_ids = tokenizer.encode(txt, add_special_tokens=True)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

        # Use a sliding window to chunk the text into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size, max_length, stride, tokenizer_name="tiktoken", shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer based on input
    if tokenizer_name == "tiktoken":
        tokenizer = tiktoken.get_encoding("cl100k") #gpt2

        #print("tokenizer tiktoken", tokenizer)  
        #print("vocab size:", tokenizer.n_vocab) #50257
    elif tokenizer_name == "sentencepiece":
        tokenizer = spm.SentencePieceProcessor(model_file="models/rilke_tokenizer.model")
        tokenizer.set_encode_extra_options("bos:eos")  # Ensure BOS/EOS tokens are used
        #print("tokenizer sentencepiece", tokenizer)  
        #print("vocab size:", tokenizer.vocab_size()) #2000
    elif tokenizer_name == "bert_base_german":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
        #print("tokenizer bert", tokenizer)  
        #print("vocab size:", tokenizer.vocab_size) #30000
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer_name}")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride, tokenizer_type=tokenizer_name)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader
