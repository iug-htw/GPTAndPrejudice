# extract_latents_simple.py
import torch
import csv

from utils.tokenization import text_to_token_ids

CSV_PATH  = "themes_analysis.csv"          # csv with columns: id, text, + label columns

def exract_latent_activations(model, sae, layer, dataset_path=CSV_PATH, device="cpu"):
    # Read CSV rows
    ids = []
    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "id" not in reader.fieldnames or "text" not in reader.fieldnames:
            raise ValueError("CSV must contain 'id' and 'text' columns.")
        for row in reader:
            ids.append(str(row["id"]))
            texts.append(row["text"])

    latents_list = []

    with torch.no_grad():
        for i, (_, sentence) in enumerate(zip(ids, texts), start=1):
            # Tokenize
            idx = text_to_token_ids(sentence)
            idx = idx.to(device)

            # Forward
            _, hidden_states, attn_weights = model(
                idx,
                output_hidden_states=True,
                output_attentions_weights=True
            )

            # Extract layer hidden & attention
            h_layer = hidden_states[layer - 1][0]      # [T, D]
            attn    = attn_weights[layer - 1][0]       # [H, T, T]

            # Attention from final token to all tokens, avg heads -> weights over tokens
            attn_to_all = attn[:, -1, :]               # [H, T]
            avg_attn_weights = attn_to_all.mean(dim=0) # [T]

            # Optional softmax to normalize (keeps original spirit but numerically safer)
            avg_attn_weights = torch.softmax(avg_attn_weights, dim=0)

            # Weighted sum of token embeddings
            weighted_hidden = torch.sum(h_layer * avg_attn_weights.unsqueeze(1), dim=0)  # [D]

            # SAE encode + top-k
            x = weighted_hidden.unsqueeze(0)
            if hasattr(sae, "decoder") and hasattr(sae.decoder, "bias"):
                x = x - sae.decoder.bias
            if hasattr(sae, "pre_encoder_bias"):
                x = x - sae.pre_encoder_bias

            z = sae.encoder(x)
            if hasattr(sae, "topk"):
                z = sae.topk(z)

            latents = z.squeeze(0)   # [M]
            latents_list.append(latents.detach().cpu())

    # Stack to [N, M] and save with ids for alignment
    latents_tensor = torch.stack(latents_list)  # [N, M]

    out = {
            "ids": ids,                  # list[str], same order as rows in CSV
            "latents": latents_tensor,   # [N, M]
            "layer": layer,
        }

    OUT_PATH  = f"sae_probing/output/latent_activations_l{layer}.pt"
    torch.save(out, OUT_PATH)

    print(f"✅ Saved {OUT_PATH} with latents shape {latents_tensor.shape} and {len(ids)} ids.")

    return out
