import torch
import json
import os
import torch.nn.functional as F
import numpy as np

from loss import calc_loss_batch
from evaluate_model import evaluate_model
from generate_text import generate

def generate_and_print_sample(model, start_context, cfg):
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        text = generate(
            model=model, prompt=start_context,
            max_new_tokens=50, context_size=cfg['context_length'],
            device="cpu",
            temperature=1,
            top_k=40,
            eos_id=13
        )
    print(text.replace("\n", " "))  # Compact print format
    model.to(cfg["device"])
    model.train()

def save_losses(train_losses, val_losses, track_tokens_seen, filename="losses.json"):
    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen
    }
    with open(filename, "w") as f:
        json.dump(data, f)


def token_repetition_loss(logits):
    """Penalize probability mass concentrating on the same tokens."""
    probs = torch.softmax(logits, dim=-1)
    rep_prob = (probs ** 2).sum(dim=-1).mean()  # Higher if repeats happen
    return rep_prob


def next_sentence_prediction_loss(hidden_states):
    """
    Compute NSP loss using mean pooled sentence embeddings and contrastive cross-entropy.
    hidden_states: list of (batch_size, seq_len, hidden_dim)
    """
    last_hidden_state = hidden_states[-1]  # Use the last layer
    sentence_embeddings = last_hidden_state.mean(dim=1)  # Mean pooling over sequence
    normalized_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
    similarity_matrix = torch.matmul(normalized_embeddings, normalized_embeddings.T)  # (batch x batch)
    targets = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
    return F.cross_entropy(similarity_matrix, targets)


def train_model_simple(
        model, train_loader, val_loader, optimizer, num_epochs,
        eval_iter, start_context, cfg, train_losses,
        val_losses, track_tokens_seen, generate_sample_text=False, 
        checkpoint_path="model_and_optimizer.pth",
        label_smoothing=0.1,
        use_lion=False):
    
    # Optional LR scheduler
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * num_epochs)

    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(f"{checkpoint_path}", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    tokens_seen, global_step = 0, -1
    device = cfg["device"]

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass - assuming model returns logits and hidden states
            logits, hidden_states = model(input_batch.to(device), output_hidden_states=True)

            # Cross-entropy with label smoothing
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_batch.view(-1).to(device),
                label_smoothing=label_smoothing
            )

            # Token repetition loss
            rep_loss = token_repetition_loss(logits)

            # Next sentence prediction loss
            nsp_loss = next_sentence_prediction_loss(hidden_states)

            # Combined loss
            total_loss = ce_loss + 0.1 * rep_loss + 0.1 * nsp_loss

            # Backward and optimize
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
#             scheduler.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation checkpoint
            if global_step % eval_iter == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device=device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                save_losses(train_losses, val_losses, track_tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Save checkpoint every epoch
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        if generate_sample_text:
            generate_and_print_sample(model, start_context, cfg)

    return train_losses, val_losses, track_tokens_seen