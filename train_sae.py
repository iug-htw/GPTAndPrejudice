import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json

def save_losses_to_json(train_losses, val_losses, filename="losses_sae.json"):
    data = {"train_losses": train_losses, "val_losses": val_losses}
    with open(filename, "w") as f:
        json.dump(data, f)

def train_sae(embeddings, sae, model_prefix="sae_model",
                 batch_size=64, epochs=50, lr=1e-4, weight_decay=1e-6,
                 device="cpu", patience=10, train_losses=[], val_losses=[]):
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

    embeddings = embeddings.to(device)
    sae.to(device)

    optimizer = optim.Adam(sae.parameters(), lr=lr, weight_decay=weight_decay)
    mse_loss = nn.MSELoss()

    dataset = TensorDataset(embeddings)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(epochs):
        sae.train()
        total_train_loss = 0
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            recon, z = sae(x)
            loss = mse_loss(recon, x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        sae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                recon, z = sae(x)
                loss = mse_loss(recon, x)
                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)

        save_losses_to_json(train_losses, val_losses, filename=f"{model_prefix}_losses.json")

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(sae.state_dict(), f"{model_prefix}.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"⏳ Early stopping at epoch {epoch+1}")
            break
        
    print(f"✅ Training complete. Best model saved as {model_prefix}.pth")

    return train_losses, val_losses