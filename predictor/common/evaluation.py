import torch
import numpy as np
from tqdm import tqdm
def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a given dataset and returns loss and accuracies."""
    model.eval()
    total_loss = 0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            all_outputs.append(outputs)
            all_labels.append(labels)

    avg_loss = total_loss / len(dataloader)
    
    # Calculate accuracies
    eval_outputs = torch.cat(all_outputs, dim=0)
    eval_labels = torch.cat(all_labels, dim=0)
    _, predicted_top10 = torch.topk(eval_outputs, 10, dim=1)
    total_samples = eval_labels.size(0)

    correct_top1 = (predicted_top10[:, 0] == eval_labels).sum().item()
    correct_top3 = (predicted_top10[:, :3] == eval_labels.unsqueeze(1)).any(dim=1).sum().item()
    correct_top5 = (predicted_top10[:, :5] == eval_labels.unsqueeze(1)).any(dim=1).sum().item()
    correct_top10 = (predicted_top10[:, :10] == eval_labels.unsqueeze(1)).any(dim=1).sum().item()

    accuracies = {
        "top1_accuracy": correct_top1 / total_samples,
        "top3_accuracy": correct_top3 / total_samples,
        "top5_accuracy": correct_top5 / total_samples,
        "top10_accuracy": correct_top10 / total_samples,
    }
    
    return avg_loss, accuracies