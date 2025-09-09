# build_neuron_concept_map.py
import os
import pandas as pd

def build_neuron_concept_map(layer=1, base_dir="sae_probing"):
    """
    For each neuron, pick the primary and secondary concept using a simple score:
      score = AP if available, else AUROC, else (P(fire|1) - P(fire|0))
    Also compute a 'polarity_score' and a categorical 'polarity_flag':
      - 'dominant'  : secondary << primary (margin big)
      - 'leaning'   : some margin
      - 'balanced'  : primary ~ secondary
      - 'two-strong': secondary >= 0.8 * primary (responds strongly to both)
    """
    path = os.path.join(base_dir, f"neuron_label_assoc_l{layer}.csv")
    df = pd.read_csv(path)

    # pick a simple score per (neuron, concept)
    def pick_score(row):
        if pd.notna(row["AP"]):      return row["AP"]
        if pd.notna(row["AUROC"]):   return row["AUROC"]
        return (row["P(fire|label=1)"] - row["P(fire|label=0)"])

    df["score"] = df.apply(pick_score, axis=1)

    rows = []
    for neuron_id, g in df.groupby("neuron_id"):
        g = g.sort_values("score", ascending=False)
        primary = g.iloc[0]
        secondary = g.iloc[1] if len(g) > 1 else None

        s1 = float(primary["score"])
        s2 = float(secondary["score"]) if secondary is not None else 0.0
        # margin ratio in [0,1]
        polarity_score = (s1 - s2) / (s1 + 1e-9)

        # simple flags
        if secondary is not None and s2 >= 0.8 * s1:
            polarity_flag = "two-strong"
        elif polarity_score >= 0.5:
            polarity_flag = "dominant"
        elif polarity_score >= 0.2:
            polarity_flag = "leaning"
        else:
            polarity_flag = "balanced"

        rows.append({
            "layer": layer,
            "neuron": int(neuron_id),

            "primary_concept": primary["concept"],
            "primary_score": round(s1, 4),
            "primary_AP": primary["AP"],
            "primary_AUROC": primary["AUROC"],
            "primary_P(fire|1)": primary["P(fire|label=1)"],
            "primary_P(fire|0)": primary["P(fire|label=0)"],
            "primary_lift": primary["lift"],
            "primary_label_support": primary["label_support"],
            "primary_neuron_fires_total": primary["neuron_fires_total"],

            "secondary_concept": (secondary["concept"] if secondary is not None else None),
            "secondary_score": (round(s2, 4) if secondary is not None else None),
            "secondary_AP": (secondary["AP"] if secondary is not None else None),
            "secondary_AUROC": (secondary["AUROC"] if secondary is not None else None),
            "secondary_P(fire|1)": (secondary["P(fire|label=1)"] if secondary is not None else None),
            "secondary_P(fire|0)": (secondary["P(fire|label=0)"] if secondary is not None else None),
            "secondary_lift": (secondary["lift"] if secondary is not None else None),
            "secondary_label_support": (secondary["label_support"] if secondary is not None else None),

            "polarity_score": round(polarity_score, 4),
            "polarity_flag": polarity_flag,
        })

    out_df = pd.DataFrame(rows).sort_values(
        ["polarity_flag", "primary_score"], ascending=[True, False]
    )

    out_path = os.path.join(base_dir, f"neuron_concept_primary_secondary_l{layer}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"✅ Saved {out_path}")
    return out_df
