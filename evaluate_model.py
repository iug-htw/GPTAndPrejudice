import torch
import torch.nn as nn

from loss import calc_loss_loader, calc_loss_loader

def evaluate_model(model, train_loader, val_loader, eval_iter, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss