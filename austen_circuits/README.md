
# Austen-GPT Circuit Discovery (IOI) — v2

A robust implementation of the **IOI circuit discovery** pipeline for your Austen GPT, with graceful fallbacks if intervention hooks aren't available.

## Features
- Build **clean vs corrupt** IOI prompts (single-token names)
- Compute **Δ logit(IO − Subject)** gap
- **Residual path patching per layer** at the prediction token (if your `GPTModel.forward` accepts `intervention_plan` and calls `plan.maybe_replace_resid_pre/post` inside each block)
- **Heuristic "name mover" head scoring** via attention mass from the prediction position to the IO token (clean run)

## Usage
```bash
python run_circuit_discovery.py --ckpt path/to/model.pth --n 256 --device cuda
```
Outputs in `out/`:
- `layer_patching_pre.csv`
- `heuristic_name_movers.csv`
- `summary.json`

## Hooking (optional but recommended)
If your `GPTModel` supports an `intervention_plan` argument and calls:
```python
x = plan.maybe_replace_resid_pre(layer_idx, x)
...
x = plan.maybe_replace_resid_post(layer_idx, x)
```
inside each transformer block, **residual patching** will work automatically.
If not, the script still runs and produces **attention-head heuristics** and baseline scores.

## Notes
- Requires `tiktoken` for GPT-2 BPE.
- If your checkpoint is a blob with `{"cfg": ..., "state_dict": ...}`, it will be loaded as-is.
  Otherwise it falls back to `GPT_CONFIG_124M` inside your `gpt_model.py`.
