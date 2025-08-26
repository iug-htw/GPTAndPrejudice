
import matplotlib.pyplot as plt

def bar_layer_restoration(results, title="Restoration by layer"):
    layers = [r.layer for r in results]
    vals   = [r.restored_mean for r in results]
    plt.figure()
    plt.bar(layers, vals)
    plt.xlabel("Layer")
    plt.ylabel("Restored Δlogit(IO−S)")
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()
