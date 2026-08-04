import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt


def save_experiment_results(metrics_dict: dict, filepath: str):
    """
    Saves test evaluation metrics to a JSON file for thesis tables.
    Supports both flat metric dicts and nested yearly metric dicts.
    """
    def sanitize(obj):
        if isinstance(obj, dict):
            return {str(k): sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return float(obj)

    clean_metrics = sanitize(metrics_dict)
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(clean_metrics, f, indent=4)
    print(f" Saved experiment results to {filepath}")


def plot_temporal_drift(
    year_results: dict, 
    title: str = "UK-LEX 18: Temporal F1 Degradation Over Time",
    save_path: str = "plots/temporal_drift_line_chart.png"
):
    """
    Plots Micro-F1, Macro-F1, Head-F1, and Tail-F1 across test set years.
    """
    years = list(year_results.keys())
    micro_f1s = [year_results[y]['Micro-F1'] for y in years]
    macro_f1s = [year_results[y]['Macro-F1'] for y in years]
    head_f1s  = [year_results[y]['Head-Micro-F1'] for y in years]
    tail_f1s  = [year_results[y]['Tail-Macro-F1'] for y in years]

    plt.figure(figsize=(10, 5))
    plt.plot(years, micro_f1s, marker='o', linewidth=2.5, label='Overall Micro-F1', color='#1f77b4')
    plt.plot(years, macro_f1s, marker='s', linewidth=2.0, label='Overall Macro-F1', color='#ff7f0e')
    plt.plot(years, head_f1s, marker='^', linestyle='--', label='Head Micro-F1', color='#2ca02c')
    plt.plot(years, tail_f1s, marker='v', linestyle='--', label='Tail Macro-F1', color='#d62728')

    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Test Set Publication Year', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f" Saved temporal drift plot to {save_path}")


@torch.no_grad()
def analyze_model_errors(
    model: torch.nn.Module, 
    dataloader, 
    raw_dataset, 
    device: torch.device, 
    label_names: list, 
    top_n: int = 5
) -> dict:
    """
    Identifies failure cases with highest prediction disagreement (False Positives / False Negatives)
    and summarizes per-class error distribution for qualitative thesis analysis.
    """
    model.eval()
    failures = []
    num_classes = len(label_names)
    class_fp_counts = np.zeros(num_classes, dtype=int)
    class_fn_counts = np.zeros(num_classes, dtype=int)

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['labels'].cpu().numpy()

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'mps' else torch.float32):
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).float().cpu().numpy()

        preds = (probs >= 0.5).astype(int)

        for i in range(len(targets)):
            true_idxs = np.where(targets[i] == 1)[0]
            pred_idxs = np.where(preds[i] == 1)[0]
            
            fp = set(pred_idxs) - set(true_idxs)
            fn = set(true_idxs) - set(pred_idxs)

            for c in fp:
                class_fp_counts[c] += 1
            for c in fn:
                class_fn_counts[c] += 1

            if len(fp) + len(fn) > 0:
                doc_idx = batch_idx * dataloader.batch_size + i
                if doc_idx < len(raw_dataset):
                    doc_meta = raw_dataset[doc_idx]
                    failures.append({
                        "doc_id": doc_meta.get('id', f"doc_{doc_idx}"),
                        "year": doc_meta.get('year', 'Unknown'),
                        "title": doc_meta.get('title', 'N/A'),
                        "true_labels": [label_names[c] for c in true_idxs],
                        "pred_labels": [label_names[c] for c in pred_idxs],
                        "error_count": len(fp) + len(fn)
                    })

    failures = sorted(failures, key=lambda x: x['error_count'], reverse=True)

    print(f"\n==================================================")
    print(f" QUALITATIVE ERROR ANALYSIS (Top {top_n} Disagreements)")
    print(f"==================================================")
    for case in failures[:top_n]:
        print(f"• ID: {case['doc_id']} | Year: {case['year']} | Title: {case['title'][:70]}...")
        print(f"  True Labels: {case['true_labels']}")
        print(f"  Predicted:   {case['pred_labels']}\n")

    return {
        "top_failures": failures[:top_n],
        "class_fp_counts": dict(zip(label_names, class_fp_counts.tolist())),
        "class_fn_counts": dict(zip(label_names, class_fn_counts.tolist()))
    }


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    head_indices,
    tail_indices,
    device: torch.device,
    epochs: int = 3,
    lr: float = 3e-5,
    checkpoint_path: str = "checkpoints/best_model.pt",
    log_interval: int = 50,
    accumulation_steps: int = 8,
    criterion=None
):
    """
    Main training loop for Legal-BERT multi-label classification with MPS memory management.
    """
    import gc
    from datetime import datetime
    from metrics import evaluate_model

    print(f"Active Device: {device}", flush=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    best_val_micro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        timestamp_start = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp_start}] Epoch {epoch}/{epochs} | Begin", flush=True)
        model.train()
        total_train_loss = 0.0
        total_batches = len(train_loader)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader, 1):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['labels'].to(device).float()

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == 'mps' else torch.float32):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, targets) / accumulation_steps

            loss.backward()
            if batch_idx % accumulation_steps == 0 or batch_idx == total_batches:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item() * accumulation_steps

            # Periodic memory cleanup every 50 batches
            if batch_idx % 50 == 0:
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()

            # Mid-epoch checkpoint every 500 batches
            if batch_idx % 500 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/latest_checkpoint.pt")
                print(f"  [{datetime.now().strftime('%H:%M:%S')}] Saved mid-epoch checkpoint at Batch {batch_idx}/{total_batches}", flush=True)

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"  [{current_time}] Epoch {epoch}/{epochs} | Batch {batch_idx}/{total_batches} | Loss: {loss.item() * accumulation_steps:.4f}", flush=True)

        avg_train_loss = total_train_loss / total_batches
        val_metrics = evaluate_model(model, val_loader, device, head_indices, tail_indices)
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"[{current_time}] --> Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Micro-F1: {val_metrics['Micro-F1']:.4f} | Val Tail-F1: {val_metrics['Tail-Macro-F1']:.4f}", flush=True)

        if val_metrics['Micro-F1'] > best_val_micro_f1:
            best_val_micro_f1 = val_metrics['Micro-F1']
            dir_name = os.path.dirname(checkpoint_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[{current_time}] Saved checkpoint to {checkpoint_path} (Val Micro-F1: {best_val_micro_f1:.4f})", flush=True)

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    print("\nTraining Complete", flush=True)
    return checkpoint_path

