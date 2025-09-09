# selective_neurons.py
import torch
import pandas as pd
import os

def load_latents(layer, base_dir=None):
    """
    Loads latent activations saved as: torch.save(latent_activations, f"latent_activations_l{layer}.pt")
    Shape expected: [num_sentences, num_latents]
    """
    latents_path = os.path.join(base_dir, f"latent_activations_l{layer}.pt")
    pack = torch.load(latents_path, map_location="cpu") 
    latents  = pack["latents"]   # [N, M]

    if latents.dim() != 2:
        raise ValueError(f"Expected 2D [N,D] latent activations, got shape {tuple(latents.shape)}")
    return latents

def find_selective_neurons(layer, min_count=5, max_count=150, activation_threshold=5.0,
                           base_dir="sae_probing"):
    """
    Select neurons that fire within [min_count, max_count] examples.
    'Firing' := (latent_value > activation_threshold). With top-k SAEs this ≈ non-zero.
    Expects a helper `load_latents(layer, base_dir)` that returns a 2D tensor [N, H].
    """
    latent_activations = load_latents(layer, base_dir=base_dir)  # [N, H]
    N, H = latent_activations.shape

    # Count activations per neuron
    activation_mask   = (latent_activations > activation_threshold)   # [N, H]
    activation_counts = activation_mask.sum(dim=0)                    # [H]

    # Windowed selectivity
    sel_mask = (activation_counts >= min_count) & (activation_counts <= max_count)
    selective_neurons = torch.nonzero(sel_mask, as_tuple=False).squeeze(1).cpu()

    # Save IDs for further analysis
    sel_pt_path = os.path.join(base_dir, f"selective_neuron_ids_l{layer}.pt")
    torch.save(selective_neurons, sel_pt_path)

    # Build DataFrame **only for selective neurons**
    counts_sel = activation_counts[sel_mask].cpu().tolist()
    ids_sel    = selective_neurons.tolist()

    df = pd.DataFrame({
        "Neuron ID": ids_sel,
        "Activation Count": counts_sel,
        "Layer": [layer] * len(ids_sel),
    }).sort_values("Activation Count", ascending=False).reset_index(drop=True)

    print(f"Layer {layer}: N={N} tokensets, H={H} neurons.")
    print(f"Window [{min_count}, {max_count}], thresh={activation_threshold}. "
          f"Selective found: {len(ids_sel)}")
    print(f"-> IDs saved to: {sel_pt_path}")

    return df

if __name__ == "__main__":
    # Example usage for your two SAEs (adjust counts if your dataset size differs)
    # Layer 6
    find_selective_neurons(layer=6, min_count=5, max_count=150, activation_threshold=0.0)
    # Layer 12
    find_selective_neurons(layer=12, min_count=5, max_count=150, activation_threshold=0.0)
