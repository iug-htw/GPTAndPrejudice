import os
import numpy as np
import pandas as pd
import torch

def top_texts_for_neuron(layer=6, neuron_id=0, top_k=10,
                         base_dir="sae_probing", csv_path="themes_analysis.csv"):
    """
    Return a DataFrame of the top-k sentences for a given neuron (highest activation first).

    Notes:
      - neuron_id is 0-based (column index in the SAE latents).
      - Uses continuous activations (no thresholding).
    """
    # Load latents
    pack = torch.load(os.path.join(base_dir, "output", f"latent_activations_l{layer}.pt"), map_location="cpu")
    ids = [str(x) for x in pack["ids"]]                    # sentence ids in row order
    latents = pack["latents"].cpu().numpy()                # shape [N, H]

    # Scores for this neuron across all sentences
    scores = latents[:, int(neuron_id)]                    # shape [N]

    # Get indices of top-k scores (descending)
    top_idx = np.argsort(-scores)[:top_k]

    # Load texts and align to the same order as ids
    df = pd.read_csv(csv_path)
    df["id"] = df["id"].astype(str)
    df = df.set_index("id").loc[ids]

    # Build result table
    out = df.iloc[top_idx].reset_index()                   # includes 'id' and 'text'
    out.insert(1, "rank", np.arange(1, len(top_idx)+1))    # 1..k
    out.insert(2, "score", scores[top_idx])                # neuron activation
    # (Optional) keep only a few columns; comment out the next line if you want all labels too
    # out = out[["id", "rank", "score", "text"]]

    return out