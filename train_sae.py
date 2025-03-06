import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import json

from sparse_auto_encoder import SparseAutoencoder

def save_losses_to_json(train_losses, val_losses, filename="losses_sae.json"):
    data = {"train_losses": train_losses, "val_losses": val_losses}
    with open(filename, "w") as f:
        json.dump(data, f)

def train_sae(embeddings, model_name="sae_model.pth", train_losses=[],
              val_losses=[], batch_size=64, epochs=100, lr=0.01,
              device="cpu", patience=10):

    if isinstance(embeddings, np.ndarray): 
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
    elif isinstance(embeddings, list):
        embeddings = torch.tensor(np.vstack(embeddings), dtype=torch.float32)

    input_dim = embeddings.shape[1]  
    hidden_dim = 256

    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    embeddings = embeddings.to(device)

    optimizer = optim.AdamW(sae.parameters(), lr=lr, weight_decay=1e-6)
    criterion = nn.MSELoss()

    dataset = TensorDataset(embeddings)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float("inf")
    early_stop_counter = 0  

    for epoch in range(epochs):
        sae.train()
        train_loss = 0.0

        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs, encoded = sae(inputs)
            loss = criterion(outputs, inputs)
            sparsity_loss = torch.norm(encoded, p=1) * 1e-4  
            total_loss_val = loss + sparsity_loss
            total_loss_val.backward()
            optimizer.step()
            train_loss += total_loss_val.item()

        sae.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs, encoded = sae(inputs)
                loss = criterion(outputs, inputs)
                sparsity_loss = torch.norm(encoded, p=1) * 1e-4
                total_loss_val = loss + sparsity_loss
                val_loss += total_loss_val.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        save_losses_to_json(train_losses, val_losses)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0  
            torch.save(sae.state_dict(), model_name)  
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"⏳ Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break  

    print(f"✅ SAE training completed. Best model saved as {model_name}.")
