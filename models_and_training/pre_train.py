import torch
import json
import os

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

def train_model_simple(
        model, train_loader, val_loader, optimizer, num_epochs,
        eval_iter, cfg, train_losses, val_losses, track_tokens_seen,
        model_prefix="model_and_optimizer",
        patience=10, min_delta=0.0):

    checkpoint_path = f"{model_prefix}.pth"

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    tokens_seen, global_step = 0, -1
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Early stopping state
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device=cfg["device"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_iter == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device=cfg["device"])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                save_losses(train_losses, val_losses, track_tokens_seen, filename=f"{model_prefix}_losses.json")
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                # Early stopping logic
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, checkpoint_path)  # Save only when improved
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered at epoch {epoch+1}.")
                        return train_losses, val_losses, track_tokens_seen

        # Still save at end of epoch (optional)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

    return train_losses, val_losses, track_tokens_seen
