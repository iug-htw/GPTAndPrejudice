import torch
import tiktoken
from gpt_model import GPTModel, DEFAULT_CFG
from utils.tokenization import text_to_token_ids, token_ids_to_text 

def generate(model, prompt, max_new_tokens, context_size=256, device="cpu", temperature=0.0, top_k=None, eos_id=None):
    tokenizer = tiktoken.get_encoding("gpt2")
    idx = text_to_token_ids(prompt, tokenizer).to(device)

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        if not isinstance(logits, tuple):
            logits = logits[0] 
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
            print(idx_next)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    ouput_text = token_ids_to_text(idx, tokenizer)
    return ouput_text

if __name__ == "__main__":
    torch.set_printoptions(profile="full")

    model = GPTModel(DEFAULT_CFG)
    model.to("cpu")

    checkpoint = torch.load("model_896_14_8_256.pth", weights_only=True, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    text = generate(
        model=model,
        prompt="The gentleman is ",
        max_new_tokens=10,
        context_size=DEFAULT_CFG['context_length'],
        device="cpu",
        temperature=0,
        top_k=10
    )

    print(text)
