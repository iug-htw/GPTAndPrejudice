# analyze_all_layers.py
import os
import pandas as pd
import numpy as np

# ==== EDIT THIS ====
ALL_LAYERS_CSV = "sae_probing/neuron_concept_primary_secondary_all_layers.csv"
OUT_DIR        = "sae_probing/analysis"
TOPK           = 10
# ===================

os.makedirs(OUT_DIR, exist_ok=True)

def load_all_layers(path):
    df = pd.read_csv(path)
    # normalize neuron column name
    if "neuron" in df.columns:
        df = df.rename(columns={"neuron": "neuron_id"})
    # keep only expected columns if present
    expected = [
        "layer","neuron_id","primary_concept","primary_AP",
        "secondary_concept","secondary_AP","polarity_score","polarity_flag"
    ]
    cols = [c for c in expected if c in df.columns]
    return df[cols].copy()

def pct(x): 
    return (100.0 * x).round(1)

def print_section(title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

def main():
    df = load_all_layers(ALL_LAYERS_CSV)
    n_rows = len(df)
    n_layers = df["layer"].nunique()
    print_section(f"Summary ({n_rows} neurons across {n_layers} layers)")
    print(df.head(3).to_string(index=False))

    # 1) Neurons per layer + growth
    print_section("Neurons per layer (growth)")
    per_layer = df.groupby("layer").size().rename("count").reset_index()
    per_layer["growth_vs_prev"] = per_layer["count"].diff().fillna(0).astype(int)
    print(per_layer.to_string(index=False))
    per_layer.to_csv(os.path.join(OUT_DIR, "neurons_per_layer.csv"), index=False)

    # 2) Mean primary AP by layer + peak layer
    print_section("Mean primary AP by layer (peak layer)")
    ap_by_layer = df.groupby("layer")["primary_AP"].mean().round(3).rename("mean_primary_AP")
    peak_layer = ap_by_layer.idxmax()
    peak_value = ap_by_layer.max()
    print(ap_by_layer.to_string())
    print(f"\nPeak layer for mean AP: L{peak_layer} (mean AP = {peak_value:.3f})")
    ap_by_layer.to_csv(os.path.join(OUT_DIR, "mean_primary_ap_by_layer.csv"))

    # 3) Polarity trends across layers
    print_section("Polarity trends across layers")
    pol_by_layer = df.groupby("layer")["polarity_score"].mean().round(3).rename("mean_polarity")
    print(pol_by_layer.to_string())
    pol_by_layer.to_csv(os.path.join(OUT_DIR, "mean_polarity_by_layer.csv"))

    # 4) Share of dominant neurons (per layer) + other flags
    print_section("Polarity flags share per layer")
    flag_counts = df.pivot_table(index="layer", columns="polarity_flag", values="neuron_id", aggfunc="count").fillna(0).astype(int)
    flag_shares = (flag_counts.div(flag_counts.sum(axis=1), axis=0)*100).round(1)
    print("Counts:\n", flag_counts.to_string())
    print("\nShares (%):\n", flag_shares.to_string())
    flag_counts.to_csv(os.path.join(OUT_DIR, "polarity_flags_counts_per_layer.csv"))
    flag_shares.to_csv(os.path.join(OUT_DIR, "polarity_flags_shares_per_layer.csv"))

    # 5) Which concepts lead (primary concept counts & strength)
    print_section("Primary concept counts & strength (mean AP)")
    primary_counts = df["primary_concept"].value_counts().rename("count")
    primary_strength = df.groupby("primary_concept")["primary_AP"].mean().round(3).rename("mean_primary_AP")
    lead_table = pd.concat([primary_counts, primary_strength], axis=1).sort_values(["count","mean_primary_AP"], ascending=[False, False])
    print(lead_table.to_string())
    lead_table.to_csv(os.path.join(OUT_DIR, "primary_concept_counts_strength.csv"))

    # 6) Entanglement: primary → secondary pairs
    print_section("Primary → Secondary pairs (top)")
    pair_counts = (
        df.assign(secondary_concept=df["secondary_concept"].fillna("None"))
          .groupby(["primary_concept","secondary_concept"])
          .size()
          .rename("count")
          .reset_index()
          .sort_values("count", ascending=False)
    )
    # share within each primary
    pair_counts["share_within_primary_%"] = (
        pair_counts.groupby("primary_concept")["count"].apply(lambda s: (s/s.sum()*100).round(1))
    )
    print(pair_counts.head(25).to_string(index=False))
    pair_counts.to_csv(os.path.join(OUT_DIR, "primary_to_secondary_pairs.csv"), index=False)

    # 7) Polarity by concept (how single-minded neurons are)
    print_section("Polarity by concept (mean polarity)")
    polarity_by_concept = df.groupby("primary_concept")["polarity_score"].mean().round(3).rename("mean_polarity")
    print(polarity_by_concept.sort_values(ascending=False).to_string())
    polarity_by_concept.to_csv(os.path.join(OUT_DIR, "polarity_by_concept.csv"))

    # 8) Strongest individual units (by AP)
    print_section(f"Top {TOPK} neurons by primary AP")
    cols_show = ["layer","neuron_id","primary_concept","primary_AP","secondary_concept","secondary_AP","polarity_score","polarity_flag"]
    top_units = df.sort_values("primary_AP", ascending=False).head(TOPK)[cols_show]
    print(top_units.to_string(index=False))
    top_units.to_csv(os.path.join(OUT_DIR, f"top_{TOPK}_neurons_by_AP.csv"), index=False)

    # 9) Count of primary concepts with no secondary
    print_section("Count of neurons with NO secondary concept")
    no_secondary = df["secondary_concept"].isna().sum()
    print(f"No-secondary neurons: {no_secondary}")
    # breakdown by layer/concept
    no_sec_by_layer = df[df["secondary_concept"].isna()].groupby("layer").size().rename("no_secondary_count")
    print("\nBy layer:\n", no_sec_by_layer.to_string())
    no_sec_by_layer.to_csv(os.path.join(OUT_DIR, "no_secondary_by_layer.csv"))
    no_sec_by_concept = df[df["secondary_concept"].isna()].groupby("primary_concept").size().rename("no_secondary_count")
    print("\nBy primary concept:\n", no_sec_by_concept.sort_values(ascending=False).to_string())
    no_sec_by_concept.to_csv(os.path.join(OUT_DIR, "no_secondary_by_concept.csv"))

    # 10) Extra: secondary concept landscape
    print_section("Secondary concept frequency overall")
    sec_counts = df["secondary_concept"].fillna("None").value_counts().rename("count_accross_all_layers")
    print(sec_counts.to_string())
    sec_counts.to_csv(os.path.join(OUT_DIR, "secondary_concept_counts.csv"))

    # 11) Extra: AP gap (primary - secondary) trends
    if "secondary_AP" in df.columns:
        print_section("AP gap (primary_AP - secondary_AP)")
        df["ap_gap"] = (df["primary_AP"] - df["secondary_AP"].fillna(0)).round(3)
        ap_gap_by_layer = df.groupby("layer")["ap_gap"].mean().round(3).rename("mean_ap_gap")
        print(ap_gap_by_layer.to_string())
        ap_gap_by_layer.to_csv(os.path.join(OUT_DIR, "mean_ap_gap_by_layer.csv"))
        ap_gap_by_concept = df.groupby("primary_concept")["ap_gap"].mean().round(3).rename("mean_ap_gap")
        print("\nBy concept:\n", ap_gap_by_concept.sort_values(ascending=False).to_string())
        ap_gap_by_concept.to_csv(os.path.join(OUT_DIR, "mean_ap_gap_by_concept.csv"))

    print_section("Done")
    print(f"Saved CSV summaries in: {OUT_DIR}")

if __name__ == "__main__":
    main()
