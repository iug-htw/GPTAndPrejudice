# saes__map_neurons_to_concepts.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_neuron_to_concept_assoc(
    layer=1,
    base_dir="sae_probing",
    csv_path="./datasets/themes_analysis.csv",
    threshold=0.0,
    eps=1e-3
):
    """
    Build the full neuron×concept associations table.
    
    For each selective neuron in a given layer, compute how strongly
    it is associated with each concept label in the probing dataset.
    
    Metrics include:
      - AP, AUROC: ranking quality based on continuous activations
      - ΔP: difference in firing probability between positives vs negatives
      - P(fire|1), P(fire|0): conditional firing probabilities
      - lift: ratio P(fire|1)/P(fire|0)
      - label_support: number of positive examples for the concept
      - neuron_fires_total: total number of sentences where neuron fired
    
    Laplace smoothing (eps) is applied to avoid divide-by-zero issues.
    
    Saves results to:
        sae_probing/output/neuron_label_assoc_l{layer}.csv
    """

    # ---- Load latent activations and selective neurons ----
    latents_pack = torch.load(
        os.path.join(base_dir, "output", f"latent_activations_l{layer}.pt"),
        map_location="cpu"
    )
    sentence_ids = [str(x) for x in latents_pack["ids"]]
    all_latents = latents_pack["latents"].cpu().numpy()  # shape [N, H]

    selective_neuron_ids = torch.load(
        os.path.join(base_dir, "output", f"selective_neuron_ids_l{layer}.pt"),
        map_location="cpu"
    ).numpy()  # [S]

    # Restrict to only the selective neurons
    all_neuron_activation_values = all_latents[:, selective_neuron_ids]         # [N, S], raw activation values
    all_neuron_binary_values  = (all_neuron_activation_values > threshold).astype(int) # [N, S], binary version of the same activations: 1 if the activation > threshold, 0 otherwise

    # ---- Load full probing dataset (sentences + concept labels) ----
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[sentence_ids]  # align to activation order

    # Collect just the label columns
    concept_labels = [c for c in df.columns if c not in ("text", "id")]
    Y = df[concept_labels].fillna(0).astype(int).values  # shape [N, C]

    # ---- Build associations ----
    rows = []
    for neuron_col_idx, neuron_id in enumerate(selective_neuron_ids):
        neuron_activation_values = all_neuron_activation_values[:, neuron_col_idx] # raw activation values for this neuron
        neuron_binary = all_neuron_binary_values[:, neuron_col_idx]    # binary version of the same activations

        for concept_idx, concept_name in enumerate(concept_labels):
            labels = Y[:, concept_idx]

            # Separate positive and negative groups
            pos_mask, neg_mask = (labels == 1), (labels == 0)
            n_pos, n_neg = pos_mask.sum(), neg_mask.sum()
            c_pos, c_neg = (neuron_binary[pos_mask] == 1).sum(), (neuron_binary[neg_mask] == 1).sum()

            # Laplace-smoothed conditional probabilities
            p_fire_given_pos = (c_pos + eps) / (n_pos + 2 * eps)
            p_fire_given_neg = (c_neg + eps) / (n_neg + 2 * eps)

            delta_p = p_fire_given_pos - p_fire_given_neg
            lift = p_fire_given_pos / (p_fire_given_neg + 1e-12)

            # Continuous ranking metrics
            try:
                ap = average_precision_score(labels, neuron_activation_values)
            except Exception:
                ap = np.nan
            try:
                auc = roc_auc_score(labels, neuron_activation_values)
            except Exception:
                auc = np.nan

            rows.append({
                "layer": layer,
                "neuron": int(neuron_id),
                "concept": concept_name,
                "AP": ap,
                "AUROC": auc,
                "ΔP": delta_p,
                "P(fire|1)": p_fire_given_pos,
                "P(fire|0)": p_fire_given_neg,
                "lift": lift,
                "label_support": int(n_pos),
                "neuron_fires_total": int(neuron_binary.sum()),
            })

    # ---- Save associations table ----
    assoc_df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, "output", f"neuron_label_assoc_l{layer}.csv")
    assoc_df.to_csv(out_path, index=False)
    print(f"✅ Associations table saved: {out_path} ({len(assoc_df)} rows)")

    return assoc_df
