import torch
import json
import os

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


def train_model_simple(
        model, train_loader, val_loader, optimizer, num_epochs,
        eval_iter, start_context, cfg, train_losses,
        val_losses, track_tokens_seen, generate_sample_text=False, 
        checkpoint_path="model_and_optimizer.pth"):
    
    # load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(f"{checkpoint_path}", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    tokens_seen, global_step = 0, -1

    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset gradients
            loss = calc_loss_batch(input_batch, target_batch, model, device=cfg["device"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step the scheduler after the optimizer step
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_iter == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter, device=cfg["device"])
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                save_losses(train_losses, val_losses, track_tokens_seen, filename="losses.json")
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, checkpoint_path)

        if generate_sample_text:
            generate_and_print_sample(model, start_context, cfg)

    return train_losses, val_losses, track_tokens_seen