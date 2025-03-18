import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import minepy
import re

def evaluate_trained_sae(sae, model, tokenizer, get_token_embeddings, layer,
                         linear_prob_label="marriage", device=torch.device("cpu")):
    # ----- Step 1: Load Text File -----
    with open('val_text_data_all_txt.txt', 'r', encoding='utf-8') as f:
        full_text = f.read()

    # ----- Step 2: Split into sentences -----
    # Simple sentence splitter (adapt as needed)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # ----- Step 3: Filter out long sentences (> 60 words) -----
    filtered_sentences = [s.strip() for s in sentences if len(s.strip().split()) <= 60]
    print(f"Total sentences after filtering: {len(filtered_sentences)}")

    # ---- Extract Hidden States ----
    hidden_states_list = []
    labels = []  # Example: 0 if neutral, 1 if marriage-related (for probing)

    for text in filtered_sentences:
        embeddings = get_token_embeddings(text, model, tokenizer, layers=[layer])
        if 6 in embeddings:
            sentence_embedding = np.mean(embeddings[layer], axis=0)
            hidden_states_list.append(sentence_embedding)

            labels.append(1 if linear_prob_label in text.lower() else 0)

    hidden_states = torch.tensor(np.array(hidden_states_list), dtype=torch.float32).to(device)
    probe_labels = torch.tensor(labels)

    # ---- 1. Reconstruction Error ----
    with torch.no_grad():
        reconstructed = sae(hidden_states)
        mse_loss = F.mse_loss(reconstructed, hidden_states)
        cosine_sim = F.cosine_similarity(reconstructed, hidden_states, dim=1).mean()

    print(f"Reconstruction MSE: {mse_loss.item():.6f}")
    print(f"Average Cosine Similarity: {cosine_sim.item():.6f}")

    # ---- 2. Linear Probe Evaluation ----
    with torch.no_grad():
        latent = sae.encode(hidden_states).cpu().numpy()
    labels = probe_labels.cpu().numpy()

    clf = LogisticRegression(max_iter=1000)
    clf.fit(latent, labels)
    preds = clf.predict(latent)
    acc = accuracy_score(labels, preds)
    print(f"Linear Probe Accuracy: {acc:.4f}")

    # ---- 3. Mutual Information (MI) ----
    mi_scores = []
    for i in range(hidden_states.shape[1]):
        mi = mutual_info_regression(latent, hidden_states[:, i].cpu().numpy(), random_state=42)
        mi_scores.append(np.mean(mi))
    print(f"Mean Mutual Information (sklearn estimate): {np.mean(mi_scores):.6f}")

    # Optional: Using minepy MIC
    m = minepy.MINE(alpha=0.6, c=15)
    mic_list = []
    for i in range(hidden_states.shape[1]):
        m.compute_score(latent[:, 0], hidden_states[:, i].cpu().numpy())
        mic_list.append(m.mic())
    print(f"Mean MIC (minepy): {np.mean(mic_list):.6f}")