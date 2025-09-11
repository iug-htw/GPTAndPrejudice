# saes__map_neurons_to_concepts.py
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

def calculate_neuron_to_concept_assoc(layer=1, base_dir="sae_probing",
                             csv_path="themes_analysis.csv",
                             threshold=0.0, eps=1e-3):
    """
    Build the full neuron×concept associations table.
    Enforces ΔP = P(fire|1) - P(fire|0) with Laplace smoothing.
    Saves CSV to: sae_probing/neuron_label_assoc_l{layer}.csv
    """
    pack = torch.load(os.path.join(base_dir, f"latent_activations_l{layer}.pt"), map_location="cpu")
    ids = [str(x) for x in pack["ids"]]
    latents = pack["latents"].cpu().numpy()
    sel = torch.load(os.path.join(base_dir, f"selective_neuron_ids_l{layer}.pt"), map_location="cpu").numpy()

    Z = latents[:, sel]                            # continuous [N,S]
    F = (Z > threshold).astype(np.int32)           # binary [N,S]

    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[ids]
    concept_cols = [c for c in df.columns if c not in ("text","id")]
    Y = df[concept_cols].fillna(0).astype(int).values

    rows = []
    for s_idx, neuron_id in enumerate(sel):
        z_cont = Z[:, s_idx]
        z_bin  = F[:, s_idx]
        for c_idx, concept in enumerate(concept_cols):
            y = Y[:, c_idx]
            pos, neg = (y==1), (y==0)
            n1, n0 = pos.sum(), neg.sum()
            c1, c0 = (z_bin[pos]==1).sum(), (z_bin[neg]==1).sum()

            # smoothed conditional probs
            p1 = (c1 + eps)/(n1 + 2*eps)
            p0 = (c0 + eps)/(n0 + 2*eps)
            dP = p1 - p0
            lift = p1/(p0+1e-12)

            try: ap = average_precision_score(y, z_cont)
            except: ap = np.nan
            try: auc = roc_auc_score(y, z_cont)
            except: auc = np.nan

            rows.append({
                "layer": layer,
                "neuron": int(neuron_id),
                "concept": concept,
                "AP": ap,
                "AUROC": auc,
                "ΔP": dP,
                "P(fire|1)": p1,
                "P(fire|0)": p0,
                "lift": lift,
                "label_support": int(n1),
                "neuron_fires_total": int(z_bin.sum()),
            })

    assoc_df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, f"neuron_label_assoc_l{layer}.csv")
    assoc_df.to_csv(out_path, index=False)
    print(f"✅ Associations table saved: {out_path} ({len(assoc_df)} rows)")
    return assoc_df