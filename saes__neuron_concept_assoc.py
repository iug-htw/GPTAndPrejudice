# saes__map_neurons_to_concepts.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_neuron_to_concept_assoc(layer=1, base_dir="sae_probing", csv_path="themes_analysis.csv",
                            threshold=5.0, topk_per_label=10):
    """
    For a given layer:
    - Load latent activations and selective neuron IDs.
    - Compute how each selective neuron relates to each concept label.
    - Save a full table and a top-K-per-label table.
    """

    # --- Load latents and selective neurons ---
    latent_pack = torch.load(os.path.join(base_dir, f"latent_activations_l{layer}.pt"), map_location="cpu")
    sentence_ids = [str(x) for x in latent_pack["ids"]]
    all_latents = latent_pack["latents"].cpu().numpy()  # shape [num_sentences, num_neurons]

    selective_neurons = torch.load(
        os.path.join(base_dir, f"selective_neuron_ids_l{layer}.pt"),
        map_location="cpu"
    ).numpy()  # indices of selective neurons

    # Select only the activations for selective neurons
    selective_latents = all_latents[:, selective_neurons]  # shape [N, S]
    selective_fires = (selective_latents > float(threshold)).astype(np.int8)  # binary version

    # --- Load CSV and align to sentence order ---
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").reindex(sentence_ids)

    # Concept label columns (all non-text columns)
    concept_labels = [c for c in df.columns if c not in ("text", "id")]
    Y = df[concept_labels].values.astype(int)   # shape [N, C]

    # --- Build neuron → concept associations ---
    associations = []
    num_sentences, num_neurons = selective_latents.shape

    for neuron_idx, neuron_id in enumerate(selective_neurons):
        neuron_scores = selective_latents[:, neuron_idx]
        neuron_fires = selective_fires[:, neuron_idx]

        for label_idx, label_name in enumerate(concept_labels):
            y = Y[:, label_idx]

            # Skip labels that are all 0 or all 1
            if y.sum() == 0 or y.sum() == len(y):
                ap, auroc, p_fire_given_1, p_fire_given_0, lift = [np.nan]*5
            else:
                p_fire_given_1 = neuron_fires[y == 1].mean() if (y == 1).any() else np.nan
                p_fire_given_0 = neuron_fires[y == 0].mean() if (y == 0).any() else np.nan
                lift = (p_fire_given_1 / (p_fire_given_0 + 1e-9)) if p_fire_given_0 == p_fire_given_0 else np.nan

                try:
                    ap = average_precision_score(y, neuron_scores)
                except Exception:
                    ap = np.nan
                try:
                    auroc = roc_auc_score(y, neuron_scores)
                except Exception:
                    auroc = np.nan

            associations.append({
                "layer": layer,
                "neuron_id": int(neuron_id),
                "concept": label_name,
                "AP": ap,
                "AUROC": auroc,
                "P(fire|label=1)": p_fire_given_1,
                "P(fire|label=0)": p_fire_given_0,
                "lift": lift,
                "label_support": int(y.sum()),
                "neuron_fires_total": int(neuron_fires.sum()),
            })

    assoc_df = pd.DataFrame(associations)

    # Save full table
    full_path = os.path.join(base_dir, f"neuron_label_assoc_l{layer}.csv")
    assoc_df.sort_values(["concept", "AP"], ascending=[True, False]).to_csv(full_path, index=False)

    # Save top-K per concept
    top_rows = []
    for label_name, group in assoc_df.groupby("concept", sort=False):
        if group["AP"].notna().any():
            group = group.sort_values("AP", ascending=False)
        else:
            group = group.sort_values("lift", ascending=False)
        top_rows.append(group.head(topk_per_label))
    top_df = pd.concat(top_rows, ignore_index=True)
    top_path = os.path.join(base_dir, f"neuron_label_top{topk_per_label}_l{layer}.csv")
    top_df.to_csv(top_path, index=False)

    print(f"✅ Saved full mapping: {full_path}")
    print(f"✅ Saved top-{topk_per_label} per concept: {top_path}")

    return top_df
