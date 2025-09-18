import torch
import torch.nn.functional as F
import numpy as np
import re

from utils.embeddings import get_token_embeddings_from_sentence

def evaluate_trained_sae(sae, model, layer, device=torch.device("cpu")):
    # ----- Step 1: Load Text File -----
    with open('val_text_data_all_txt.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # ----- Step 2: Split into sentences -----
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # ----- Step 3: Filter out long sentences (> 60 words) -----
    filtered_sentences = [s.strip() for s in sentences if len(s.strip().split()) <= 60]
    print(f"Total sentences after filtering: {len(filtered_sentences)}")

    # ---- Extract Hidden States ----
    hidden_states_list = []

    for text in filtered_sentences:
        embeddings = get_token_embeddings_from_sentence(text, model, device=device)[layer]
        if layer in embeddings:
            sentence_embedding = np.mean(embeddings[layer], axis=0)
            hidden_states_list.append(sentence_embedding)

    hidden_states = torch.tensor(np.array(hidden_states_list), dtype=torch.float32).to(device)

    # ---- 1. Reconstruction Error ----
    with torch.no_grad():
        reconstructed, latent = sae(hidden_states)  # Unpack tuple
        mse_loss = F.mse_loss(reconstructed, hidden_states)
        cosine_sim = F.cosine_similarity(reconstructed, hidden_states, dim=1).mean()

    print(f"Reconstruction MSE: {mse_loss.item():.6f}")
    print(f"Average Cosine Similarity: {cosine_sim.item():.6f}")

    # ---- 2. L0 Sparsity Metric (Average active latents per sample) ----
    active_counts = (latent.abs() > 1e-5).sum(dim=1)  # Count non-zero latents
    avg_l0_sparsity = active_counts.float().mean().item()
    print(f"Average L0 Sparsity (active latents): {avg_l0_sparsity:.2f}")

    # ---- 3. Cross-Entropy / KL Loss Increase Simulation ----
    # Dummy linear next-token predictor simulating downstream impact
    downstream_head = torch.nn.Linear(hidden_states.shape[1], 100).to(device)  # Assume 100 tokens vocab size
    torch.nn.init.normal_(downstream_head.weight, std=0.02)

    with torch.no_grad():
        original_logits = downstream_head(hidden_states)
        reconstructed_logits = downstream_head(reconstructed)

        ce_loss_original = F.cross_entropy(original_logits, original_logits.argmax(dim=1))
        ce_loss_reconstructed = F.cross_entropy(reconstructed_logits, original_logits.argmax(dim=1))
        kl_loss = F.kl_div(F.log_softmax(reconstructed_logits, dim=1),
                           F.softmax(original_logits, dim=1),
                           reduction='batchmean')

    print(f"Cross-Entropy Loss (original): {ce_loss_original.item():.6f}")
    print(f"Cross-Entropy Loss (reconstructed): {ce_loss_reconstructed.item():.6f}")
    print(f"KL Divergence (Reconstructed || Original): {kl_loss.item():.6f}")