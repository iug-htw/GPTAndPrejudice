import torch
import json
import argparse
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

matplotlib.use("Agg")  

def load_losses(filename="losses.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    if "track_tokens_seen" in data:
        track_tokens_seen = data["track_tokens_seen"]
    else:
        track_tokens_seen = []
    return data["train_losses"], data["val_losses"], track_tokens_seen

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    if len(tokens_seen) > 0:
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot-sea-12-1536.pdf")
    # plt.show() # disabling for the HPC

if __name__ == "__main__":
    num_epochs = 30

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_prefix", type=str, required=True, help="Prefix for the model and loss files", default="model_and_optimizer")

    args = parser.parse_args()
    train_losses, val_losses, track_tokens_seen = load_losses(args.model_prefix + "_losses.json")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, track_tokens_seen, train_losses, val_losses)