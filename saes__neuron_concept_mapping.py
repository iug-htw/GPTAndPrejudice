# saes__neuron_concept_map_signed.py
import os, numpy as np, pandas as pd

def build_neuron_concept_map(layer=1, base_dir="sae_probing"):
    """
    Read neuron_label_assoc_l{layer}.csv and pick primary/secondary concepts per neuron.
    Saves to: sae_probing/neuron_concept_primary_secondary_l{layer}.csv
    """
    path = os.path.join(base_dir, f"neuron_label_assoc_l{layer}.csv")
    df = pd.read_csv(path)

    rows = []
    for neuron_id, g in df.groupby("neuron"):
        # Only keep positive ΔP associations
        g = g[g["ΔP"] > 0].copy()
        if g.empty: continue
        g = g.sort_values("AP", ascending=False)

        primary = g.iloc[0]
        secondary = g.iloc[1] if len(g) > 1 else None

        s1 = primary["AP"]
        s2 = secondary["AP"] if secondary is not None else 0.0
        polarity_score = (s1 - s2)/(s1 + 1e-9)

        if secondary is not None and s2 >= 0.8*s1:
            flag = "two-strong"
        elif polarity_score >= 0.5:
            flag = "dominant"
        elif polarity_score >= 0.2:
            flag = "leaning"
        else:
            flag = "balanced"

        rows.append({
            "layer": layer,
            "neuron": int(neuron_id),
            "primary_concept": primary["concept"],
            "primary_AP": s1,
            "secondary_concept": secondary["concept"] if secondary is not None else None,
            "secondary_AP": s2 if secondary is not None else None,
            "polarity_score": polarity_score,
            "polarity_flag": flag
        })

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(base_dir, f"neuron_concept_primary_secondary_l{layer}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Primary/secondary mapping saved: {out_path}")
    return out_df
