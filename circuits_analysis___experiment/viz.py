
"""viz.py
Matplotlib plotting helpers for circuit analysis.
Rules: matplotlib only, one chart per function, no explicit color choices.
"""
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

def head_importance_heatmap(matrix: np.ndarray, title: str = "Head importance (gain)", xlabel: str = "Head", ylabel: str = "Layer", savepath: Optional[str] = None):
    """matrix shape: [n_layers, n_heads]"""
    fig = plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, aspect='auto')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()

def layer_importance_bar(values: List[float], title: str = "Layer patch gain", xlabel: str = "Layer", ylabel: str = "Gain", savepath: Optional[str] = None):
    idx = np.arange(len(values))
    fig = plt.figure(figsize=(8, 4))
    plt.bar(idx, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()

def token_logit_diff_trend(diffs: List[float], title: str = "Logit diff across steps", xlabel: str = "Step", ylabel: str = "Logit diff", savepath: Optional[str] = None):
    idx = np.arange(len(diffs))
    fig = plt.figure(figsize=(8, 4))
    plt.plot(idx, diffs)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
