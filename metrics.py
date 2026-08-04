import torch
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def compute_temporal_f1_metrics(
    y_true: np.ndarray, 
    y_pred_binary: np.ndarray, 
    head_indices: list, 
    tail_indices: list
) -> dict:
    """
    Computes overall Micro/Macro F1 alongside partitioned Head/Tail Micro/Macro F1 scores.
    """
    micro_f1 = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)

    # Convert head_indices and tail_indices to 1D integer NumPy arrays for NumPy advanced indexing
    if isinstance(head_indices, (torch.Tensor, np.ndarray)):
        head_indices = head_indices.cpu().numpy() if isinstance(head_indices, torch.Tensor) else head_indices
        head_indices = head_indices.astype(int)
    else:
        head_indices = np.array([int(x) for x in head_indices], dtype=int)

    if isinstance(tail_indices, (torch.Tensor, np.ndarray)):
        tail_indices = tail_indices.cpu().numpy() if isinstance(tail_indices, torch.Tensor) else tail_indices
        tail_indices = tail_indices.astype(int)
    else:
        tail_indices = np.array([int(x) for x in tail_indices], dtype=int)

    # Head and Tail subsets
    head_micro = f1_score(y_true[:, head_indices], y_pred_binary[:, head_indices], average='micro', zero_division=0)
    head_macro = f1_score(y_true[:, head_indices], y_pred_binary[:, head_indices], average='macro', zero_division=0)

    tail_micro = f1_score(y_true[:, tail_indices], y_pred_binary[:, tail_indices], average='micro', zero_division=0)
    tail_macro = f1_score(y_true[:, tail_indices], y_pred_binary[:, tail_indices], average='macro', zero_division=0)

    return {
        "Micro-F1": micro_f1,
        "Macro-F1": macro_f1,
        "Head-Micro-F1": head_micro,
        "Head-Macro-F1": head_macro,
        "Tail-Micro-F1": tail_micro,
        "Tail-Macro-F1": tail_macro,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "head_micro_f1": head_micro,
        "head_macro_f1": head_macro,
        "tail_micro_f1": tail_micro,
        "tail_macro_f1": tail_macro
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device, 
    head_indices: list, 
    tail_indices: list
) -> dict:
    """
    Evaluation loop for validation and test splits with progress bar.
    """
    model.to(device)
    model.eval()
    all_preds, all_targets = [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].float().cpu().numpy()

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'mps' else torch.float32):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits).float().cpu().numpy()

        all_preds.append((probs >= 0.5).astype(int))
        all_targets.append(targets)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    return compute_temporal_f1_metrics(y_true, y_pred, head_indices, tail_indices)


def evaluate_by_year(
    model: torch.nn.Module, 
    dataset, 
    device: torch.device, 
    head_indices: list, 
    tail_indices: list, 
    collate_fn=None, 
    batch_size: int = 32
) -> dict:
    """
    Year-by-year evaluation engine for temporal concept drift assessment.
    """
    if collate_fn is None:
        from dataset import get_data_collator
        collate_fn = get_data_collator()

    model.eval()
    years_arr = np.array([int(y) for y in dataset['year']])
    unique_years = sorted(list(set(years_arr)))
    year_results = {}
    
    for yr in unique_years:
        indices = np.where(years_arr == yr)[0]
        if len(indices) == 0:
            continue

        sub_ds = dataset.select(indices)
        sub_loader = DataLoader(sub_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        metrics = evaluate_model(model, sub_loader, device, head_indices, tail_indices)
        year_results[int(yr)] = metrics
        print(f"  Evaluated Year {int(yr)}: {len(indices)} documents")
        
    return year_results
