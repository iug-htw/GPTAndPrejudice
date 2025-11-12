import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def run_sae_tpp_test(sae, embeddings, labels, target_label, top_k=5, device=torch.device("cpu")):
    """
    SAE Targeted Probe Perturbation (TPP)-like evaluation.
    
    Args:
        sae: Trained SAE model (must output (reconstruction, latent))
        embeddings: torch.Tensor [N, D], LLM hidden states embeddings
        labels: np.array of binary labels (1 if "marriage & female", 0 otherwise)
        target_label: str, description of the concept being probed (for clarity)
        top_k: int, number of latents to ablate
        device: torch device

    Returns:
        dict with baseline accuracy, ablated accuracy, accuracy drop, and attribution scores
    """
    sae = sae.to(device)
    embeddings = embeddings.to(device)

    # Pass through SAE to get latent activations
    with torch.no_grad():
        _, latent = sae(embeddings)  # Shape: [N, latent_dim]

    latent_np = latent.cpu().numpy()

    # Step 1: Train linear probe on the latent space
    clf = LogisticRegression(max_iter=1000)
    clf.fit(latent_np, labels)
    probe_weights = torch.tensor(clf.coef_[0], dtype=torch.float32).to(device)

    # Step 2: Compute attribution score for each latent
    a_pos = latent[labels == 1].mean(dim=0)
    a_neg = latent[labels == 0].mean(dim=0)
    attribution_scores = (probe_weights * (a_pos - a_neg)).cpu().numpy()

    # Step 3: Identify top_k most contributing latents
    top_latents_idx = np.argsort(np.abs(attribution_scores))[-top_k:]

    # Step 4: Baseline probe accuracy
    preds = clf.predict(latent_np)
    baseline_acc = accuracy_score(labels, preds)

    # Step 5: Zero-out top-k latents (ablation)
    latent_ablate = latent.clone()
    latent_ablate[:, top_latents_idx] = 0
    latent_ablate_np = latent_ablate.cpu().numpy()

    # Step 6: Re-run probe on ablated latents
    ablated_preds = clf.predict(latent_ablate_np)
    ablated_acc = accuracy_score(labels, ablated_preds)

    # Step 7: Report
    result = {
        "concept": target_label,
        "baseline_accuracy": baseline_acc,
        "ablated_accuracy": ablated_acc,
        "accuracy_drop": baseline_acc - ablated_acc,
        "top_latents_idx": top_latents_idx.tolist(),
        "attribution_scores": attribution_scores[top_latents_idx].tolist()
    }

    print(f"[TPP Test] Concept: {target_label}")
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Ablated Accuracy:  {ablated_acc:.4f}")
    print(f"Accuracy Drop:     {baseline_acc - ablated_acc:.4f}")
    print(f"Top-{top_k} Latents: {top_latents_idx.tolist()}")

    return result
