import torch
import json
import os
import math
from torch.optim.lr_scheduler import OneCycleLR

from loss import calc_loss_batch
from evaluate_model import evaluate_model

def save_losses(train_losses, val_losses, track_tokens_seen, filename="losses.json"):
    data = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "track_tokens_seen": track_tokens_seen
    }
    with open(filename, "w") as f:
        json.dump(data, f)

def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """
    Linear warmup to base LR for `warmup_steps`, then cosine decay to min_lr_ratio * base LR.
    Works with multiple param groups (uses each group's initial lr internally).
    """
    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine phase
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # scale between 1.0 → min_lr_ratio
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_model_simple(
        model, train_loader, val_loader, optimizer, num_epochs,
        eval_iter, cfg, train_losses, val_losses, track_tokens_seen,
        model_prefix="model_and_optimizer",
        warmup_steps=1500,          # ← NEW: warmup steps
        min_lr_ratio=0.0            # ← NEW: cosine target (0.0 = decay to zero)
    ):
    """
    Trains with linear warmup followed by cosine decay.
    Persists scheduler state for proper resume.
    """
    checkpoint_path = f"{model_prefix}.pth"

    max_lr = 3e-4
    tokens_seen, global_step = 0, -1
    total_steps = len(train_loader) * num_epochs
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=0.06,            # ~6% warmup
        anneal_strategy='cos',
        div_factor=10,             # start at max_lr/10
        final_div_factor=10        # end at ~max_lr/10
    )

    # Optional resume
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Try to load scheduler state if present (keeps LR phase aligned)
        if "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as e:
                print(f"[warn] Could not load scheduler state: {e}")

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            loss = calc_loss_batch(input_batch, target_batch, model, device=cfg["device"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # <- warmup + cosine

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_iter == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device=cfg["device"])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                save_losses(train_losses, val_losses, track_tokens_seen, filename=f"{model_prefix}_losses.json")
                # Show current LR from first param group for reference
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, LR {current_lr:.6f}")

        # Save checkpoint (now includes scheduler)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, checkpoint_path)

    return train_losses, val_losses, track_tokens_seen
