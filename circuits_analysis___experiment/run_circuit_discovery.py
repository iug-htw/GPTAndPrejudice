#!/usr/bin/env python3
import os, re, csv, json, argparse
import torch
import tiktoken

# ----------------------------
# Imports for your project
# ----------------------------
# Try both import paths so this script works whether you run from project root
# or inside the circuits_analysis___experiment package folder.
from .data.ioi_data import build_ioi_dataset
from gpt_model import GPTModel, DEFAULT_CFG

# ----------------------------
# Helper: tokenizer
# ----------------------------
def _enc():
    return tiktoken.get_encoding("gpt2")

# ----------------------------
# Pad + adjust target positions
# ----------------------------
def pad_and_stack_with_adjusted_positions(batch_token_ids, orig_positions, pad_id: int = 50256):
    """
    Left-pad sequences to a common length and shift each position by its pad offset.

    Returns:
      ids_padded  [B, T]
      pos_adjusted list[int] (len B)
    """
    T = max(len(x) for x in batch_token_ids)
    B = len(batch_token_ids)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    new_pos = []
    for i, ids in enumerate(batch_token_ids):
        offset = T - len(ids)          # number of pads on the left
        out[i, offset:] = torch.tensor(ids, dtype=torch.long)
        new_pos.append(offset + orig_positions[i])
    return out, new_pos

# ----------------------------
# Metrics
# ----------------------------
def io_logit_diff(logits: torch.Tensor, target_pos, io_ids, s_ids) -> torch.Tensor:
    """
    Δlogit = logit(IO) - logit(S) at the prediction slot (target_pos).
    """
    B, T, V = logits.shape
    device = logits.device
    idx = torch.arange(B, device=device)
    tp  = torch.tensor(target_pos, device=device)
    io  = torch.tensor(io_ids, device=device)
    sj  = torch.tensor(s_ids, device=device)
    lt = logits[idx, tp, :]  # [B,V]
    return lt[idx, io] - lt[idx, sj]

def mean_io_logit_diff(logits, target_pos, io_ids, s_ids) -> float:
    return io_logit_diff(logits, target_pos, io_ids, s_ids).mean().item()

# ----------------------------
# Model I/O + flexible loader
# ----------------------------
def _cfg_from_filename(path: str):
    """
    Try to parse: model_{emb}_{heads}_{layers}_{ctx}.pth
    e.g., model_896_14_8_256.pth -> emb=896, n_heads=14, n_layers=8, context_length=256
    """
    m = re.search(r"(\d+)_(\d+)_(\d+)_(\d+)", os.path.basename(path))
    if not m:
        return None
    emb, heads, layers, ctx = map(int, m.groups())
    cfg = dict(DEFAULT_CFG)  # shallow copy
    cfg.update({
        "emb_dim": emb,
        "n_heads": heads,
        "n_layers": layers,
        "context_length": ctx,
    })
    return cfg

def load_model(ckpt_path: str, device: str = "cpu"):
    blob = torch.load(ckpt_path, map_location=device)

    # Case A: training checkpoint dict with model_state_dict
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state = blob["model_state_dict"]
        cfg = blob.get("cfg") or _cfg_from_filename(ckpt_path) or DEFAULT_CFG
        model = GPTModel(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    # Case B: { "cfg": ..., "state_dict": ... }
    if isinstance(blob, dict) and "cfg" in blob and "state_dict" in blob:
        model = GPTModel(blob["cfg"]).to(device)
        model.load_state_dict(blob["state_dict"], strict=True)
        model.eval()
        return model

    # Case C: plain state_dict
    if isinstance(blob, dict):
        cfg = _cfg_from_filename(ckpt_path) or DEFAULT_CFG
        model = GPTModel(cfg).to(device)
        model.load_state_dict(blob, strict=True)
        model.eval()
        return model

    raise RuntimeError("Unrecognized checkpoint format.")

# ----------------------------
# Forward helper (robust output unpacking)
# ----------------------------
def run_model(model, ids, collect_attn: bool = True):
    """
    Calls model.forward with different signatures gracefully.
    Returns: logits, cache (dict or None), attn (list[tensor] or None)
    """
    out = model.forward(ids, enable_cache=True, output_attentions_weights=collect_attn)
    if isinstance(out, tuple):
        if len(out) == 4:
            logits, cache, hidden, attn = out
        elif len(out) == 3:
            logits, cache, hidden = out; attn = None
        elif len(out) == 2:
            logits, cache = out; hidden = None; attn = None
        else:
            logits = out[0]; cache = None; attn = None
    else:
        logits, cache, hidden, attn = out, None, None, None
    return logits, cache, attn

# ----------------------------
# Residual patching (chunked)
# ----------------------------
class ResidualPatcherPlan:
    """
    Minimal intervention plan that can swap resid_pre/resid_post
    with clean-cache values at selected positions.
    """
    def __init__(self, clean_cache):
        self.clean_cache = clean_cache  # dict: key -> layer -> [B,T,d]
        self.instructions = {}          # layer -> (mode, mask[B,T], clean[B,T,d])

    def add_patch(self, layer_idx: int, mode: str, mask: torch.Tensor):
        key = "resid_pre" if mode == "pre" else "resid_post"
        clean_tensor = self.clean_cache[key][layer_idx]  # [B,T,d]
        self.instructions[layer_idx] = (mode, mask.clone(), clean_tensor.clone())

    # Hooks expected to be called by your GPTModel blocks:
    def maybe_replace_resid_pre(self, layer_idx, x: torch.Tensor) -> torch.Tensor:
        return self._apply(layer_idx, x, "pre")

    def maybe_replace_resid_post(self, layer_idx, x: torch.Tensor) -> torch.Tensor:
        return self._apply(layer_idx, x, "post")

    def _apply(self, layer_idx: int, x: torch.Tensor, mode: str) -> torch.Tensor:
        instr = self.instructions.get(layer_idx)
        if instr is None:
            return x
        m, mask, clean = instr
        if m != mode:
            return x
        return torch.where(mask.unsqueeze(-1).to(x.device), clean.to(x.device), x)

def make_mask_for_positions(batch_size: int, seq_len: int, positions):
    m = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    for i in range(batch_size):
        tp = positions[i]
        if 0 <= tp < seq_len:
            m[i, tp] = True
    return m

def layerwise_resid_patching_chunked(model, clean_cache, ids_corrupt,
                                     target_positions, io_ids, s_ids,
                                     mode="pre", chunk=64):
    """
    Memory-friendly: process corrupt batch in chunks, slicing the clean cache to match.
    Requires your model's blocks to call plan.maybe_replace_resid_pre/post.
    """
    device = next(model.parameters()).device
    B, T = ids_corrupt.shape
    base_logits, _, _ = run_model(model, ids_corrupt, collect_attn=False)
    base = mean_io_logit_diff(base_logits, target_positions, io_ids, s_ids)

    # Num layers
    n_layers = getattr(model, "n_layers", None) or len(getattr(model, "trf_blocks", [])) or 12
    totals = [0.0] * n_layers
    counts = [0]   * n_layers

    for start in range(0, B, chunk):
        end = min(B, start + chunk)
        ids_chunk = ids_corrupt[start:end]
        tp_chunk  = target_positions[start:end]
        io_chunk  = io_ids[start:end]
        s_chunk   = s_ids[start:end]

        # Slice clean cache to this chunk
        clean_chunk = {"resid_pre": {}, "resid_post": {}}
        for key in ("resid_pre", "resid_post"):
            if key in clean_cache:
                for L, tens in clean_cache[key].items():
                    clean_chunk[key][L] = tens[start:end].to(device, non_blocking=True)

        Bc, Tc = ids_chunk.shape
        mask = make_mask_for_positions(Bc, Tc, tp_chunk)

        for L in range(n_layers):
            plan = ResidualPatcherPlan(clean_chunk)
            plan.add_patch(L, mode, mask)
            try:
                out = model.forward(ids_chunk, intervention_plan=plan)
            except TypeError:
                # Intervention not supported
                continue
            logits_p = out[0] if isinstance(out, tuple) else out
            score = mean_io_logit_diff(logits_p, tp_chunk, io_chunk, s_chunk)
            totals[L] += (score - base) * (end - start)
            counts[L] += (end - start)

    results = []
    for L in range(n_layers):
        restored = (totals[L] / counts[L]) if counts[L] else 0.0
        results.append({"layer": L, "mode": mode, "restored_mean": restored})
    return results

# ----------------------------
# Logging helpers (CSV)
# ----------------------------
def _decode_ids(ids_1d):
    return _enc().decode(list(map(int, ids_1d)))

def _decode_tok(tok_id: int):
    return _enc().decode([int(tok_id)])

def _gather_next_logits(logits: torch.Tensor, target_positions):
    B, T, V = logits.shape
    idx = torch.arange(B, device=logits.device)
    tp  = torch.tensor(target_positions, device=logits.device)
    return logits[idx, tp, :]

def _topk_tokens(logits_row, k=5):
    vals, ids = torch.topk(logits_row, k)
    out = []
    for j in range(k):
        tid = int(ids[j].item())
        out.append((tid, _decode_tok(tid), float(vals[j].item())))
    return out

def _rows_for_logging(variant, ids_batch, logits, target_positions, io_ids, s_ids, pad_id=50256, topk=5):
    """
    Build rows where prompt_text ENDS AT THE SLOT (exclude the IO token).
    Assumes target_positions are pad-adjusted.
    """
    B, T = ids_batch.shape
    next_logits = _gather_next_logits(logits, target_positions)  # [B,V]
    rows = []
    for b in range(B):
        seq = [int(x) for x in ids_batch[b].tolist()]
        # trim left padding
        start = 0
        while start < T and seq[start] == pad_id:
            start += 1
        tp = int(target_positions[b])
        if tp <= start or tp >= T:
            # malformed row; skip
            continue

        # prompt ends at the slot (just after "to ")
        prefix = seq[start:tp]
        prompt_text = _enc().decode(prefix)

        logits_row = next_logits[b]
        pred_id = int(torch.argmax(logits_row).item())
        pred_str = _decode_tok(pred_id)

        io_id = int(io_ids[b]); s_id = int(s_ids[b])
        io_str = _decode_tok(io_id); s_str = _decode_tok(s_id)

        logit_pred = float(logits_row[pred_id].item())
        logit_io   = float(logits_row[io_id].item())
        logit_s    = float(logits_row[s_id].item())

        topk_list = _topk_tokens(logits_row, k=topk)
        topk_ids    = [t[0] for t in topk_list]
        topk_tokens = [t[1] for t in topk_list]
        topk_vals   = [t[2] for t in topk_list]

        rows.append({
            "variant": variant,
            "b_idx": b,
            "target_pos": tp,
            "prompt_text": prompt_text,
            "pred_token": pred_str,
            "pred_id": pred_id,
            "completed_preview": prompt_text + pred_str,
            "io_token": io_str,
            "io_id": io_id,
            "s_token": s_str,
            "s_id": s_id,
            "logit_pred": f"{logit_pred:.6f}",
            "logit_io": f"{logit_io:.6f}",
            "logit_s": f"{logit_s:.6f}",
            "topk_ids": "|".join(map(str, topk_ids)),
            "topk_tokens": "|".join(topk_tokens),
            "topk_logits": "|".join(f"{x:.4f}" for x in topk_vals),
            "is_pred_io": int(pred_id == io_id),
            "is_pred_subject": int(pred_id == s_id),
        })
    return rows

def _write_csv(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("empty\n")
        return
    keys = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _merge_clean_corrupt(clean_rows, corrupt_rows):
    by_b_clean = {r["b_idx"]: r for r in clean_rows}
    by_b_corr  = {r["b_idx"]: r for r in corrupt_rows}
    merged = []
    for b in sorted(set(by_b_clean) & set(by_b_corr)):
        c = by_b_clean[b]; k = by_b_corr[b]
        merged.append({
            "b_idx": b,
            "clean_prompt": c["prompt_text"],
            "clean_pred": c["pred_token"],
            "clean_io": c["io_token"],
            "clean_logit_io": c["logit_io"],
            "clean_logit_s": c["logit_s"],
            "corrupt_prompt": k["prompt_text"],
            "corrupt_pred": k["pred_token"],
            "corrupt_io": k["io_token"],
            "corrupt_logit_io": k["logit_io"],
            "corrupt_logit_s": k["logit_s"],
        })
    return merged

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to model checkpoint (.pth)")
    ap.add_argument("--n", type=int, default=256, help="Number of IOI pairs")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--no-attn", action="store_true", help="Do not collect attention weights")
    ap.add_argument("--skip-patching", action="store_true", help="Skip residual path patching step")
    ap.add_argument("--patch-chunk", type=int, default=64, help="Chunk size for patching to limit memory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load model
    model = load_model(args.ckpt, args.device)
    pad_id = (getattr(model, "cfg", {}) or {}).get("pad_token_id", 50256)

    # 2) Build dataset
    data = build_ioi_dataset(n=args.n, seed=0)

    # 3) Pad inputs + adjust target positions
    ids_clean,   tp_clean   = pad_and_stack_with_adjusted_positions(
        data.token_ids_clean, data.target_positions_clean, pad_id=pad_id
    )
    ids_corrupt, tp_corrupt = pad_and_stack_with_adjusted_positions(
        data.token_ids_corrupt, data.target_positions_corrupt, pad_id=pad_id
    )
    ids_clean   = ids_clean.to(args.device)
    ids_corrupt = ids_corrupt.to(args.device)

    # Sanity: ensure the slot is truly “… to {NAME}”
    enc = _enc()
    for i in range(min(5, len(tp_clean))):
        prev = enc.decode([int(ids_clean[i, tp_clean[i]-1])]).strip()
        if prev != "to":
            raise AssertionError(f"row {i}: expected 'to' before IO, got '{prev}'")

    # 4) Forward passes
    collect_attn = not args.no_attn
    clean_logits, clean_cache, clean_attn = run_model(model, ids_clean,  collect_attn=collect_attn)
    corrupt_logits, corrupt_cache, _      = run_model(model, ids_corrupt, collect_attn=False)

    # 5) Metrics
    clean_score   = mean_io_logit_diff(clean_logits,   tp_clean,   data.io_token_ids, data.s_token_ids)
    corrupt_score = mean_io_logit_diff(corrupt_logits, tp_corrupt, data.io_token_ids, data.s_token_ids)
    gap = clean_score - corrupt_score

    print(f"Clean mean Δlogit(IO−S):   {clean_score:.4f}")
    print(f"Corrupt mean Δlogit(IO−S): {corrupt_score:.4f}")
    print(f"Gap (clean − corrupt):     {gap:.4f}")

    # 6) Logging CSVs (predictions)
    clean_rows = _rows_for_logging(
        variant="clean",
        ids_batch=ids_clean,
        logits=clean_logits,
        target_positions=tp_clean,
        io_ids=data.io_token_ids,
        s_ids=data.s_token_ids,
        pad_id=pad_id,
        topk=5,
    )
    corrupt_rows = _rows_for_logging(
        variant="corrupt",
        ids_batch=ids_corrupt,
        logits=corrupt_logits,
        target_positions=tp_corrupt,
        io_ids=data.io_token_ids,
        s_ids=data.s_token_ids,
        pad_id=pad_id,
        topk=5,
    )
    _write_csv(clean_rows,   os.path.join(args.outdir, "predictions_clean.csv"))
    _write_csv(corrupt_rows, os.path.join(args.outdir, "predictions_corrupt.csv"))
    merged = _merge_clean_corrupt(clean_rows, corrupt_rows)
    _write_csv(merged, os.path.join(args.outdir, "predictions_merged.csv"))
    print(f"[log] wrote {len(clean_rows)} clean rows, {len(corrupt_rows)} corrupt rows, {len(merged)} merged")

    # 7) Residual patching (optional & chunked)
    patch_results = []
    if not args.skip_patching:
        try:
            patch_results = layerwise_resid_patching_chunked(
                model, clean_cache, ids_corrupt,
                tp_corrupt, data.io_token_ids, data.s_token_ids,
                mode="pre", chunk=args.patch_chunk
            )
        except Exception as e:
            print(f"[WARN] Residual patching not supported or failed: {e}")

    # Save patch results
    with open(os.path.join(args.outdir, "layer_patching_pre.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["layer", "mode", "restored_mean"])
        for r in patch_results:
            w.writerow([r["layer"], r["mode"], f"{r['restored_mean']:.6f}"])

    # 8) Heuristic name mover heads (if attention available)
    head_scores = []
    if collect_attn and clean_attn:
        # Expect clean_attn as list of [B, H, T_q, T_k]
        ids_clean_cpu = ids_clean.cpu()
        io_positions = []
        for b in range(ids_clean_cpu.shape[0]):
            tok = data.io_token_ids[b]
            # Find last occurrence of IO token in the padded row
            pos = -1
            for t in range(ids_clean_cpu.shape[1]-1, -1, -1):
                if int(ids_clean_cpu[b, t].item()) == int(tok):
                    pos = t; break
            io_positions.append(pos)

        for L, attn in enumerate(clean_attn):
            if attn.dim() != 4:  # [B, H, T_q, T_k]
                continue
            H = attn.size(1)
            for h in range(H):
                a = attn[:, h, :, :]
                total = 0.0; count = 0
                for b in range(ids_clean_cpu.shape[0]):
                    tp = tp_clean[b]; ip = io_positions[b]
                    if tp < 0 or ip < 0 or tp >= a.size(1) or ip >= a.size(2):
                        continue
                    total += float(a[b, tp, ip].item()); count += 1
                if count:
                    head_scores.append((L, h, total / count))

        head_scores.sort(key=lambda x: x[2], reverse=True)
        with open(os.path.join(args.outdir, "heuristic_name_movers.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["layer", "head", "mean_attention_mass_on_IO"])
            for (L, h, s) in head_scores[:100]:
                w.writerow([L, h, f"{s:.6f}"])

    # 9) Summary JSON
    summary = {
        "clean_mean_logit_diff": clean_score,
        "corrupt_mean_logit_diff": corrupt_score,
        "gap": gap,
        "top_layer_restorations": sorted(
            [{"layer": r["layer"], "restored_mean": r["restored_mean"]} for r in patch_results],
            key=lambda x: x["restored_mean"], reverse=True
        )[:10],
        "top_heuristic_heads": head_scores[:10] if head_scores else []
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.outdir}/layer_patching_pre.csv, "
          f"{args.outdir}/heuristic_name_movers.csv, predictions_*.csv, and summary.json")

if __name__ == "__main__":
    main()
