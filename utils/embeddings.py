import torch
from utils.tokenization import text_to_token_ids

def _extract_embeddings(text, model, device):
    """
    Extracts token embeddings from specified transformer layers.

    Args:
    - text (str): Input text.
    - model: Custom GPT model.
    - tokenizer: tiktoken encoding object.
    - layers (list): Transformer layers to extract embeddings from.

    Returns:
    - dict: Layer-wise token embeddings {layer_number: embeddings}
    """

    input_ids = text_to_token_ids(text).to(device)

    with torch.no_grad():
        _, hidden_states = model(input_ids, output_hidden_states=True)

    return [state.squeeze(0).cpu().numpy() for state in hidden_states]

def get_token_embeddings_from_dataset(dataset, model, device="cpu"):
    layers_embeddings = {
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: [],
        8: [],
        9: [],
        10: [],
        11: [],
        12: [],
    }

    for sentence in dataset:
        embeddings = _extract_embeddings(sentence, model, device)
        for i in range(12):
            layers_embeddings[i + 1].append(embeddings[i])

    return layers_embeddings


def get_token_embeddings_from_sentence(text, model, device="cpu"):
    layers_embeddings = {
        1: None,
        2: None,
        3: None,
        4: None,
        5: None,
        6: None,
        7: None,
        8: None,
        9: None,
        10: None,
        11: None,
        12: None,
    }

    embeddings = _extract_embeddings(text, model, device)
    print('===>', len(embeddings), i)
    for i in range(12):
        layers_embeddings[i + 1] = embeddings[i]

    return layers_embeddings