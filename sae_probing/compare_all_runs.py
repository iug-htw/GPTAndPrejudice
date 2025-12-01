import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, chisquare
from sklearn.metrics import mutual_info_score
from scipy.special import rel_entr

ROOT_DIR = "sae_probing"
OUT_METRICS = os.path.join(ROOT_DIR, "run_metrics_summary.csv")
OUT_DOMINANT = os.path.join(ROOT_DIR, "run_dominant_overlap_summary.csv")

# how many dominant items to consider
TOPK_DOM_CONCEPTS = 25
TOPK_DOM_PAIRS = 50


def bootstrap_ci(values, n_boot=5000, ci=95):
    values = np.asarray(values)
    if len(values) == 0:
        return np.nan, np.nan
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return lower, upper


# ---------- helpers to derive dominant tables ----------

def _compute_primary_from_mapping(mapping_df: pd.DataFrame):
    """Fallback: compute primary concept counts & mean AP from mapping."""
    if mapping_df is None or mapping_df.empty:
        return None
    grp = (
        mapping_df
        .groupby("primary_concept", dropna=True)
        .agg(
            count=("primary_concept", "size"),
            mean_primary_AP=("primary_AP", "mean"),
        )
        .reset_index()
    )
    return grp


def _compute_pairs_from_mapping(mapping_df: pd.DataFrame):
    """Fallback: compute primary->secondary pair counts from mapping."""
    if mapping_df is None or mapping_df.empty:
        return None
    df = mapping_df.copy()
    df["secondary_concept"] = df["secondary_concept"].fillna("None")
    grp = (
        df.groupby(["primary_concept", "secondary_concept"])
          .size()
          .reset_index(name="count")
    )
    return grp


def _load_primary_concepts(run_dir, mapping_df):
    path = os.path.join(run_dir, "analysis", "primary_concept_counts_strength.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if "primary_concept" not in df.columns:
            df = df.rename(columns={df.columns[0]: "primary_concept"})
        return df
    # fallback: derive from mapping
    return _compute_primary_from_mapping(mapping_df)


def _load_primary_secondary_pairs(run_dir, mapping_df):
    path = os.path.join(run_dir, "analysis", "primary_to_secondary_pairs.csv")
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if "primary_concept" in df.columns and "secondary_concept" in df.columns:
            return df
    # fallback: derive from mapping
    return _compute_pairs_from_mapping(mapping_df)


def load_runs(root=ROOT_DIR):
    """
    Returns:
        run_name -> {
            "mapping": mapping_df,
            "primary": primary_concepts_df or None,
            "pairs": primary_secondary_pairs_df or None,
        }
    """
    runs = {}
    for name in os.listdir(root):
        run_dir = os.path.join(root, name)
        mapping_path = os.path.join(
            run_dir,
            "mappings",
            "neuron_concept_primary_secondary_all_layers.csv",
        )
        if os.path.isfile(mapping_path):
            try:
                mapping_df = pd.read_csv(mapping_path)

                if mapping_df.empty:
                    print(f"{name}: mapping is empty, dominant info will be None")
                    primary_df = None
                    pairs_df = None
                else:
                    primary_df = _load_primary_concepts(run_dir, mapping_df)
                    pairs_df = _load_primary_secondary_pairs(run_dir, mapping_df)

                runs[name] = {
                    "mapping": mapping_df,
                    "primary": primary_df,
                    "pairs": pairs_df,
                }
                print(f"Loaded run: {name} ({len(mapping_df)} rows)")
            except Exception as e:
                print(f"Failed to load {mapping_path}: {e}")
    return runs


def get_id_cols(df):
    cols = ["layer"]
    if "neuron" in df.columns:
        cols.append("neuron")
    elif "neuron_id" in df.columns:
        cols.append("neuron_id")
    else:
        raise ValueError("Expected a 'neuron' or 'neuron_id' column.")
    return cols


def align_mapping_dfs(A, B):
    A = A.copy()
    B = B.copy()
    if "neuron_id" in A.columns and "neuron" in B.columns:
        B = B.rename(columns={"neuron": "neuron_id"})
    elif "neuron" in A.columns and "neuron_id" in B.columns:
        B = B.rename(columns={"neuron_id": "neuron"})
    id_cols = get_id_cols(A)
    return A, B, id_cols


def compare_two_runs(nameA, runA, nameB, runB):
    print(f"\n=== Comparing {nameA} vs {nameB} ===")

    A = runA["mapping"]
    B = runB["mapping"]

    A_aligned, B_aligned, id_cols = align_mapping_dfs(A, B)

    merged = A_aligned.merge(
        B_aligned,
        on=id_cols,
        how="inner",
        suffixes=("_A", "_B"),
    )

    if merged.empty:
        print("No overlapping neurons; skipping comparison.")
        metrics_row = {
            "run_A": nameA,
            "run_B": nameB,
            "n_overlap_neurons": 0,
        }
        dom_row = {
            "run_A": nameA,
            "run_B": nameB,
            # overlap stats
            "dom_concepts_k": np.nan,
            "dom_concepts_intersection": np.nan,
            "dom_concepts_union": np.nan,
            "dom_concepts_jaccard": np.nan,
            "dom_concepts_recall_A_in_B": np.nan,
            "dom_concepts_recall_B_in_A": np.nan,
            "dom_pairs_k": np.nan,
            "dom_pairs_intersection": np.nan,
            "dom_pairs_union": np.nan,
            "dom_pairs_jaccard": np.nan,
            "dom_pairs_recall_A_in_B": np.nan,
            "dom_pairs_recall_B_in_A": np.nan,
            # explicit lists
            "top_concepts_A": np.nan,
            "top_concepts_B": np.nan,
            "top_pairs_A": np.nan,
            "top_pairs_B": np.nan,
        }
        return metrics_row, dom_row

    # -------------------- METRICS SUMMARY --------------------
    metrics = {
        "run_A": nameA,
        "run_B": nameB,
        "n_overlap_neurons": len(merged),
        "n_neurons_A": len(A),
        "n_neurons_B": len(B),
    }

    # 1. AP comparison
    apA = merged["primary_AP_A"]
    apB = merged["primary_AP_B"]
    diff = apB - apA

    metrics["ap_mean_diff_B_minus_A"] = diff.mean()

    try:
        w = wilcoxon(apA, apB)
        metrics["ap_wilcoxon_stat"] = w.statistic
        metrics["ap_wilcoxon_p"] = w.pvalue
    except Exception as e:
        print("AP Wilcoxon failed:", e)
        metrics["ap_wilcoxon_stat"] = np.nan
        metrics["ap_wilcoxon_p"] = np.nan

    try:
        t = ttest_rel(apA, apB)
        metrics["ap_ttest_stat"] = t.statistic
        metrics["ap_ttest_p"] = t.pvalue
    except Exception as e:
        print("AP t-test failed:", e)
        metrics["ap_ttest_stat"] = np.nan
        metrics["ap_ttest_p"] = np.nan

    if diff.std() > 0:
        metrics["ap_cohen_d"] = diff.mean() / diff.std()
    else:
        metrics["ap_cohen_d"] = 0.0

    ci_low, ci_high = bootstrap_ci(diff.values)
    metrics["ap_ci_low"] = ci_low
    metrics["ap_ci_high"] = ci_high

    # 2. Polarity comparison
    if "polarity_score_A" in merged.columns and "polarity_score_B" in merged.columns:
        polA = merged["polarity_score_A"]
        polB = merged["polarity_score_B"]
        pol_diff = polB - polA
        metrics["pol_mean_diff_B_minus_A"] = pol_diff.mean()
        try:
            wpol = wilcoxon(polA, polB)
            metrics["pol_wilcoxon_stat"] = wpol.statistic
            metrics["pol_wilcoxon_p"] = wpol.pvalue
        except Exception as e:
            print("Polarity Wilcoxon failed:", e)
            metrics["pol_wilcoxon_stat"] = np.nan
            metrics["pol_wilcoxon_p"] = np.nan
    else:
        metrics["pol_mean_diff_B_minus_A"] = np.nan
        metrics["pol_wilcoxon_stat"] = np.nan
        metrics["pol_wilcoxon_p"] = np.nan

    # 3. Primary concept distribution shift
    freqA = A["primary_concept"].value_counts().sort_index()
    freqB = B["primary_concept"].value_counts().sort_index()
    all_labels = sorted(set(freqA.index) | set(freqB.index))
    freqA = freqA.reindex(all_labels, fill_value=0)
    freqB = freqB.reindex(all_labels, fill_value=0)

    pA_concepts = freqA / freqA.sum()
    pB_concepts = freqB / freqB.sum()
    eps = 1e-12
    total = 10000

    chi = chisquare(
        f_obs=(pB_concepts + eps) * total,
        f_exp=(pA_concepts + eps) * total,
    )
    metrics["concepts_chi2_stat"] = chi.statistic
    metrics["concepts_chi2_p"] = chi.pvalue

    kl = np.sum(rel_entr(pA_concepts + eps, pB_concepts + eps))
    metrics["concepts_kl_div"] = kl

    # 4. Primary -> secondary entanglement shift (MI)
    pairsA = A.assign(
        secondary_concept=A["secondary_concept"].fillna("None")
    )[["primary_concept", "secondary_concept"]]
    pairsB = B.assign(
        secondary_concept=B["secondary_concept"].fillna("None")
    )[["primary_concept", "secondary_concept"]]

    codesA = pd.factorize(list(zip(pairsA["primary_concept"], pairsA["secondary_concept"])))[0]
    codesB = pd.factorize(list(zip(pairsB["primary_concept"], pairsB["secondary_concept"])))[0]

    miA = mutual_info_score(codesA, codesA)
    miB = mutual_info_score(codesB, codesB)
    metrics["mi_primary_secondary_A"] = miA
    metrics["mi_primary_secondary_B"] = miB
    metrics["mi_primary_secondary_diff_B_minus_A"] = miB - miA

    # 5. No-secondary proportion
    nA_no_sec = A["secondary_concept"].isna().sum()
    nB_no_sec = B["secondary_concept"].isna().sum()
    N_A = len(A)
    N_B = len(B)
    pA_no_sec = nA_no_sec / N_A if N_A > 0 else np.nan
    pB_no_sec = nB_no_sec / N_B if N_B > 0 else np.nan

    metrics["prop_no_secondary_A"] = pA_no_sec
    metrics["prop_no_secondary_B"] = pB_no_sec
    metrics["prop_no_secondary_diff_B_minus_A"] = pB_no_sec - pA_no_sec

    # -------------------- DOMINANT OVERLAP + EXPLICIT LISTS --------------------
    dom = {
        "run_A": nameA,
        "run_B": nameB,
    }

    domA = runA["primary"]
    domB = runB["primary"]

    if domA is not None and domB is not None:
        sort_cols = []
        if "count" in domA.columns:
            sort_cols.append("count")
        if "mean_primary_AP" in domA.columns:
            sort_cols.append("mean_primary_AP")
        if not sort_cols:
            sort_cols = [domA.columns[0]]

        topA = domA.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(TOPK_DOM_CONCEPTS)
        topB = domB.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(TOPK_DOM_CONCEPTS)

        setA = set(topA["primary_concept"])
        setB = set(topB["primary_concept"])

        inter = setA & setB
        union = setA | setB

        dom["dom_concepts_k"] = TOPK_DOM_CONCEPTS
        dom["dom_concepts_intersection"] = len(inter)
        dom["dom_concepts_union"] = len(union)
        dom["dom_concepts_jaccard"] = len(inter) / len(union) if len(union) > 0 else np.nan
        dom["dom_concepts_recall_A_in_B"] = len(inter) / len(setA) if len(setA) > 0 else np.nan
        dom["dom_concepts_recall_B_in_A"] = len(inter) / len(setB) if len(setB) > 0 else np.nan

        # explicit ranked lists of concepts for each run
        dom["top_concepts_A"] = ",".join(topA["primary_concept"].astype(str).tolist())
        dom["top_concepts_B"] = ",".join(topB["primary_concept"].astype(str).tolist())
    else:
        dom["dom_concepts_k"] = np.nan
        dom["dom_concepts_intersection"] = np.nan
        dom["dom_concepts_union"] = np.nan
        dom["dom_concepts_jaccard"] = np.nan
        dom["dom_concepts_recall_A_in_B"] = np.nan
        dom["dom_concepts_recall_B_in_A"] = np.nan
        dom["top_concepts_A"] = np.nan
        dom["top_concepts_B"] = np.nan

    # dominant primary->secondary pairs
    pairsA_ana = runA["pairs"]
    pairsB_ana = runB["pairs"]

    if pairsA_ana is not None and pairsB_ana is not None:
        if "count" in pairsA_ana.columns:
            sort_cols_pairs = ["count"]
        else:
            sort_cols_pairs = [pairsA_ana.columns[-1]]

        topPairsA = pairsA_ana.sort_values(sort_cols_pairs, ascending=False).head(TOPK_DOM_PAIRS)
        topPairsB = pairsB_ana.sort_values(sort_cols_pairs, ascending=False).head(TOPK_DOM_PAIRS)

        setPairsA = set(zip(topPairsA["primary_concept"], topPairsA["secondary_concept"]))
        setPairsB = set(zip(topPairsB["primary_concept"], topPairsB["secondary_concept"]))

        inter_p = setPairsA & setPairsB
        union_p = setPairsA | setPairsB

        dom["dom_pairs_k"] = TOPK_DOM_PAIRS
        dom["dom_pairs_intersection"] = len(inter_p)
        dom["dom_pairs_union"] = len(union_p)
        dom["dom_pairs_jaccard"] = len(inter_p) / len(union_p) if len(union_p) > 0 else np.nan
        dom["dom_pairs_recall_A_in_B"] = len(inter_p) / len(setPairsA) if len(setPairsA) > 0 else np.nan
        dom["dom_pairs_recall_B_in_A"] = len(inter_p) / len(setPairsB) if len(setPairsB) > 0 else np.nan

        # explicit ranked lists of pairs for each run
        fmtA = (topPairsA["primary_concept"].astype(str)
                + "→"
                + topPairsA["secondary_concept"].astype(str))
        fmtB = (topPairsB["primary_concept"].astype(str)
                + "→"
                + topPairsB["secondary_concept"].astype(str))
        dom["top_pairs_A"] = ",".join(fmtA.tolist())
        dom["top_pairs_B"] = ",".join(fmtB.tolist())
    else:
        dom["dom_pairs_k"] = np.nan
        dom["dom_pairs_intersection"] = np.nan
        dom["dom_pairs_union"] = np.nan
        dom["dom_pairs_jaccard"] = np.nan
        dom["dom_pairs_recall_A_in_B"] = np.nan
        dom["dom_pairs_recall_B_in_A"] = np.nan
        dom["top_pairs_A"] = np.nan
        dom["top_pairs_B"] = np.nan

    return metrics, dom


def compare_all_runs():
    print("Scanning runs in", ROOT_DIR)
    runs = load_runs(ROOT_DIR)
    run_names = sorted(runs.keys())

    print("\nFound runs:", run_names)

    metric_rows = []
    dom_rows = []

    for i in range(len(run_names)):
        for j in range(i + 1, len(run_names)):
            nameA = run_names[i]
            nameB = run_names[j]
            metrics_row, dom_row = compare_two_runs(nameA, runs[nameA], nameB, runs[nameB])
            metric_rows.append(metrics_row)
            dom_rows.append(dom_row)

    if metric_rows:
        df_metrics = pd.DataFrame(metric_rows)
        df_metrics.to_csv(OUT_METRICS, index=False)
        print("Saved metrics summary to:", OUT_METRICS)
    else:
        print("No metric comparisons were computed.")

    if dom_rows:
        df_dom = pd.DataFrame(dom_rows)
        df_dom.to_csv(OUT_DOMINANT, index=False)
        print("Saved dominant-overlap summary to:", OUT_DOMINANT)
    else:
        print("No dominant-overlap comparisons were computed.")


if __name__ == "__main__":
    compare_all_runs()
