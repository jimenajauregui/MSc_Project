import os
import requests
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding


def get_head_tail_indices(train_targets: torch.Tensor, num_classes: int):
    """
    Ranks labels strictly by training frequency to split into 50/50 Head vs. Tail subsets.
    Prevents temporal data leakage by deriving splits strictly from historical training data.
    """
    if not isinstance(train_targets, torch.Tensor):
        train_targets = torch.tensor(np.array(train_targets), dtype=torch.float32)

    train_label_counts = torch.sum(train_targets, dim=0).cpu().numpy().astype(int)
    sorted_indices = np.argsort(train_label_counts)[::-1]  # Descending order

    num_head = num_classes // 2
    head_indices = [int(x) for x in sorted(list(sorted_indices[:num_head]))]
    tail_indices = [int(x) for x in sorted(list(sorted_indices[num_head:]))]

    return head_indices, tail_indices


def load_uklex18_dataset(
    file_path: str = 'data/uk-lex18.jsonl',
    tokenizer_name: str = 'nlpaueb/legal-bert-base-uncased',
    max_length: int = 384
):
    """
    Loads and tokenizes the UK-LEX-18 chronological dataset splits:
    - Train (<= 2002), Val (2003-007), Test (2008-2018)
    """
    url = 'https://zenodo.org/records/6355465/files/uk-lex18.jsonl?download=1'
    if not os.path.exists(file_path):
        print(f"Downloading {file_path}...")
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            f.write(response.content)

    raw_ds = load_dataset('json', data_files=file_path)
    df_uk = raw_ds['train'].to_pandas()
    df_uk['year'] = pd.to_numeric(df_uk['year'], errors='coerce')

    uk_train = df_uk[df_uk['year'] <= 2002].copy()
    uk_val   = df_uk[(df_uk['year'] > 2002) & (df_uk['year'] <= 2007)].copy()
    uk_test  = df_uk[df_uk['year'] >= 2008].copy()

    split_dict = DatasetDict({
        'train': Dataset.from_pandas(uk_train.reset_index(drop=True)),
        'validation': Dataset.from_pandas(uk_val.reset_index(drop=True)),
        'test': Dataset.from_pandas(uk_test.reset_index(drop=True))
    })

    # Build unique label index mapping
    all_uk_labels = set()
    for labels_list in df_uk['labels']:
        if isinstance(labels_list, (list, np.ndarray)):
            for l in labels_list:
                all_uk_labels.add(str(l).strip())

    sorted_uk_labels = sorted(list(all_uk_labels))
    uk_label_to_idx = {label: idx for idx, label in enumerate(sorted_uk_labels)}
    num_uk_labels = len(sorted_uk_labels)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def process_uk_batch(batch):
        tokenized = tokenizer(
            batch['body'],
            truncation=True,
            max_length=max_length
        )

        batch_encoded_vectors = []
        for labels_list in batch['labels']:
            encoded_vector = np.zeros(num_uk_labels, dtype=np.float32)
            if isinstance(labels_list, (list, np.ndarray)):
                for label in labels_list:
                    label_str = str(label).strip()
                    if label_str in uk_label_to_idx:
                        idx = uk_label_to_idx[label_str]
                        if idx < num_uk_labels:
                            encoded_vector[idx] = 1.0
            batch_encoded_vectors.append(encoded_vector)

        tokenized['labels'] = batch_encoded_vectors
        return tokenized

    tokenized_split = split_dict.map(
        process_uk_batch,
        batched=True,
        batch_size=1000,
        remove_columns=['id', 'data_type']
    )

    cols_to_format = ['input_ids', 'attention_mask', 'labels']
    if 'year' in tokenized_split['train'].column_names:
        cols_to_format.append('year')

    tokenized_split.set_format(type='torch', columns=cols_to_format)

    # Compute Head and Tail indices
    raw_train_labels = tokenized_split['train']['labels']
    head_indices, tail_indices = get_head_tail_indices(raw_train_labels, num_uk_labels)

    return tokenized_split, uk_label_to_idx, head_indices, tail_indices


def load_eurlex21_dataset(
    tokenizer_name: str = 'nlpaueb/legal-bert-base-uncased',
    max_length: int = 384
):
    """
    Loads and tokenizes the MultiEURLEX (English) Level 1 chronological dataset splits:
    - Train (1958-2010), Val (2010-2012), Test (2012-2016)
    """
    parquet_branch = "hf://datasets/coastalcph/multi_eurlex@refs/convert/parquet"
    data_files = {
        "train": f"{parquet_branch}/en/train/*.parquet",
        "validation": f"{parquet_branch}/en/validation/*.parquet",
        "test": f"{parquet_branch}/en/test/*.parquet",
    }

    eurlex_ds = load_dataset("parquet", data_files=data_files)
    num_classes = 21
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def preprocess_eurlex_batch(batch):
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

        batch_multi_hot = []
        for label_ids in batch["labels"]:
            multi_hot = [0.0] * num_classes
            for label_id in label_ids:
                if 0 <= label_id < num_classes:
                    multi_hot[label_id] = 1.0
            batch_multi_hot.append(multi_hot)

        encoded["labels"] = batch_multi_hot
        encoded["year"] = [int(c[1:5]) if len(c) >= 5 and c[1:5].isdigit() else 2012 for c in batch["celex_id"]]
        return encoded

    cols_to_remove = [col for col in eurlex_ds["train"].column_names if col not in ["input_ids", "attention_mask", "labels", "year", "celex_id", "text", "title"]]

    encoded_ds = eurlex_ds.map(
        preprocess_eurlex_batch,
        batched=True,
        batch_size=1000,
        remove_columns=cols_to_remove
    )

    cols_to_format = ["input_ids", "attention_mask", "labels", "year"]
    encoded_ds.set_format(type="torch", columns=cols_to_format)

    # Compute Head and Tail indices
    raw_train_labels = encoded_ds['train']['labels']
    head_indices, tail_indices = get_head_tail_indices(raw_train_labels, num_classes)

    return encoded_ds, head_indices, tail_indices



def get_data_collator(tokenizer_name: str = 'nlpaueb/legal-bert-base-uncased'):
    """
    Returns a dynamic padding data collator for fast batching.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return DataCollatorWithPadding(tokenizer=tokenizer)
