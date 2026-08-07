import os
import re
import sys
import json
import torch
import requests
import numpy as np
from typing import List, Dict, Optional
from tqdm.auto import tqdm
from metrics import compute_temporal_f1_metrics


class LlamaPromptBuilder:
    """
    Prompt engineering template engine for zero-shot and few-shot multi-label
    legal text classification using LLM Decoders (Llama-3-8B-Instruct).
    """
    def __init__(self, candidate_labels: List[str], domain_name: str = "EUR-LEX EuroVoc"):
        self.candidate_labels = sorted(candidate_labels)
        self.domain_name = domain_name
        self.label_to_idx = {label: idx for idx, label in enumerate(self.candidate_labels)}

    def build_zero_shot_prompt(self, document_text: str, document_title: str = "") -> str:
        """
        Constructs a zero-shot instruction prompt forcing strict JSON array output.
        """
        formatted_labels = "\n".join([f" - {label}" for label in self.candidate_labels])
        
        prompt = f"""You are an expert legal text classifier analyzing {self.domain_name} statutory legislation.

Your task is to assign all relevant legal subject categories to the document below.

Candidate Legal Categories:
{formatted_labels}

Document Title: {document_title if document_title else 'N/A'}
Document Text:
\"\"\"
{document_text[:1500]}
\"\"\"

Instructions:
1. Select ONLY applicable categories from the exact list of Candidate Legal Categories provided above.
2. Return your answer strictly as a JSON list of strings. Do not include any explanations, markdown code blocks, or extra text.

JSON Output:"""
        return prompt

    def build_few_shot_prompt(self, document_text: str, exemplar_docs: List[Dict], document_title: str = "", tokenizer=None, text_length: int = 3500) -> str:
        """
        Constructs an optimized few-shot instruction prompt using official Llama-3 Chat Markup
        (<|start_header_id|>system/user/assistant<|end_header_id|>) and in-context legal exemplars.
        """
        formatted_labels = "\n".join([f" - {label}" for label in self.candidate_labels])
        
        system_message = (
            f"You are an expert legal text classifier analyzing {self.domain_name} statutory legislation.\n"
            "Your task is to assign all applicable legal subject categories to legal documents.\n\n"
            f"Candidate Legal Categories:\n{formatted_labels}\n\n"
            "Instructions:\n"
            "1. Select ONLY applicable categories from the exact list of Candidate Legal Categories above.\n"
            "2. Return your answer strictly as a JSON list of strings (e.g. [\"CATEGORY_1\", \"CATEGORY_2\"]). Do not include any explanations, code blocks, or extra text."
        )

        messages = [{"role": "system", "content": system_message}]

        # Add Few-Shot Exemplars as user/assistant conversational turns
        for ex in exemplar_docs:
            ex_title = ex.get('title', 'N/A')
            ex_text = ex.get('text', ex.get('body', ''))[:800]
            ex_labels = ex.get('labels', [])
            
            if isinstance(ex_labels, torch.Tensor):
                ex_labels = ex_labels.cpu().numpy()
                
            ex_label_strs = []
            if isinstance(ex_labels, (np.ndarray, list)) and len(ex_labels) > 0:
                if len(ex_labels) == len(self.candidate_labels):
                    ex_label_strs = [self.candidate_labels[i] for i, val in enumerate(ex_labels) if float(val) > 0.5 and i < len(self.candidate_labels)]
                elif isinstance(ex_labels[0], (str, bytes)):
                    ex_label_strs = [str(lbl).strip() for lbl in ex_labels]
                else:
                    ex_label_strs = [self.candidate_labels[int(i)] for i in ex_labels if int(i) < len(self.candidate_labels)]

            user_text = f"Document Title: {ex_title}\nDocument Text:\n\"\"\"\n{ex_text}\n\"\"\""
            assistant_text = json.dumps(ex_label_strs)
            
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        # Target Document
        target_user_text = f"Classify the following document:\nDocument Title: {document_title if document_title else 'N/A'}\nDocument Text:\n\"\"\"\n{document_text[:text_length]}\n\"\"\"\n\nJSON Output:"
        messages.append({"role": "user", "content": target_user_text})

        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        # Fallback string
        prompt_parts = [system_message, "\n--- Few-Shot Exemplars ---"]
        for m in messages[1:]:
            prompt_parts.append(f"{m['role'].upper()}: {m['content']}")
        return "\n\n".join(prompt_parts)


def parse_llm_json_output(llm_output_text: str, candidate_labels: List[str]) -> np.ndarray:
    """
    Parses LLM output text, extracts JSON label arrays, and converts to a multi-hot binary vector.
    """
    num_classes = len(candidate_labels)
    multi_hot = np.zeros(num_classes, dtype=np.float32)
    label_to_idx = {label.upper().strip(): idx for idx, label in enumerate(candidate_labels)}

    # Attempt to extract JSON array using regex
    json_match = re.search(r'\[.*?\]', llm_output_text, re.DOTALL)
    predicted_labels = []

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                predicted_labels = [str(item).upper().strip() for item in parsed]
        except Exception:
            pass

    # Fallback substring matching if JSON parsing failed
    if not predicted_labels:
        for label_name in candidate_labels:
            if label_name.upper() in llm_output_text.upper():
                predicted_labels.append(label_name.upper())

    # Map matched label strings to binary vector
    for label_str in predicted_labels:
        if label_str in label_to_idx:
            idx = label_to_idx[label_str]
            multi_hot[idx] = 1.0

    return multi_hot


def evaluate_llama3_zero_shot(
    prompts: List[str], 
    dataset_split, 
    candidate_labels: List[str], 
    head_indices: list, 
    tail_indices: list, 
    dataset_name: str = "UK-LEX-18", 
    max_samples: Optional[int] = None,
    checkpoint_dir: str = "results",
    save_every: int = 100,
    batch_size: int = 16,
    max_new_tokens: int = 50,
    llama_model = None,
    tokenizer = None
):
    """
    Executes Llama-3 zero-shot inference with partial checkpoints and auto-resume.
    Supports batched GPU execution on CUDA (Colab Pro) for ultra-fast generation,
    with local sequential Ollama fallback.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    clean_name = dataset_name.lower().replace("-", "")
    ckpt_path = os.path.join(checkpoint_dir, f"llama3_preds_{clean_name}_ckpt.npy")

    if max_samples is not None:
        prompts = prompts[:max_samples]
        if hasattr(dataset_split, 'select'):
            dataset_split = dataset_split.select(range(min(max_samples, len(dataset_split))))
        elif isinstance(dataset_split, dict):
            dataset_split = {k: v[:max_samples] for k, v in dataset_split.items()}

    # Check for existing checkpoint
    all_pred_vectors = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        try:
            saved_preds = np.load(ckpt_path)
            all_pred_vectors = list(saved_preds)
            start_idx = len(all_pred_vectors)
            print(f" Resuming {dataset_name} evaluation from checkpoint at doc {start_idx:,}/{len(prompts):,}...")
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {ckpt_path} ({e}). Starting fresh.")
            all_pred_vectors = []
            start_idx = 0

    if start_idx < len(prompts):
        if start_idx == 0:
            print(f"\n Starting Llama-3 Zero-Shot on {dataset_name} Test Set ({len(prompts):,} documents)...")
        
        # Check notebook main scope and global scope for PyTorch CUDA model if not explicitly passed
        if llama_model is None:
            if 'llama_model' in globals():
                llama_model = globals()['llama_model']
            elif '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'llama_model'):
                llama_model = getattr(sys.modules['__main__'], 'llama_model')

        if tokenizer is None:
            if 'tokenizer' in globals():
                tokenizer = globals()['tokenizer']
            elif '__main__' in sys.modules and hasattr(sys.modules['__main__'], 'tokenizer'):
                tokenizer = getattr(sys.modules['__main__'], 'tokenizer')

        # Batched PyTorch GPU Inference (Colab Pro / Cloud GPU)
        if llama_model is not None and tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            print(f" Running batched PyTorch CUDA inference (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
            pbar = tqdm(total=len(prompts), initial=start_idx, desc=f"Llama-3 GPU ({dataset_name})")
            
            for b_start in range(start_idx, len(prompts), batch_size):
                b_end = min(b_start + batch_size, len(prompts))
                batch_prompts = prompts[b_start:b_end]
                
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=2048
                ).to("cuda")
                
                with torch.no_grad():
                    outputs = llama_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                input_len = inputs.input_ids.shape[1]
                for seq in outputs:
                    gen_tokens = seq[input_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    pred_vector = parse_llm_json_output(gen_text, candidate_labels)
                    all_pred_vectors.append(pred_vector)
                
                # Clear CUDA cache to prevent memory fragmentation
                del inputs, outputs
                torch.cuda.empty_cache()
                
                pbar.update(b_end - b_start)
                
                # Save partial checkpoint
                if len(all_pred_vectors) % save_every < batch_size or b_end == len(prompts):
                    np.save(ckpt_path, np.array(all_pred_vectors, dtype=np.float32))
            pbar.close()

        else:
            # Check if running in GPU environment
            if torch.cuda.is_available() or 'google.colab' in sys.modules:
                raise ValueError(
                    "'llama_model' or 'tokenizer' is None!\n"
                    "Please make sure you have run Cell 34 ('4.2. Llama-3-8B 4-Bit Model Initialization') first "
                    "to load the Llama-3 model into GPU memory, and pass llama_model=llama_model, tokenizer=tokenizer."
                )
            
            # Sequential Ollama Fallback for local CPU testing
            print(f" Running local sequential Ollama inference...")
            for i in tqdm(range(start_idx, len(prompts)), desc=f"Llama-3 Ollama ({dataset_name})", initial=start_idx, total=len(prompts)):
                prompt = prompts[i]
                res = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3:8b", "prompt": prompt, "stream": False}
                )
                generated_text = res.json().get("response", "") if res.status_code == 200 else ""
                pred_vector = parse_llm_json_output(generated_text, candidate_labels)
                all_pred_vectors.append(pred_vector)
                
                if (i + 1) % save_every == 0 or (i + 1) == len(prompts):
                    np.save(ckpt_path, np.array(all_pred_vectors, dtype=np.float32))

    y_pred = np.array(all_pred_vectors, dtype=np.float32)
    y_true = dataset_split['labels']
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    else:
        y_true = np.array(y_true, dtype=np.float32)
        
    # Calculate Temporal Metrics (y_true first, y_pred second)
    metrics = compute_temporal_f1_metrics(y_true, y_pred, head_indices, tail_indices)
    
    print("\n" + "="*60)
    print(f"{dataset_name} ZERO-SHOT LLAMA-3 TEST RESULTS")
    print("="*60)
    print(f"  • Micro-F1       : {metrics.get('micro_f1', metrics.get('Micro-F1', 0.0)):.4f}")
    print(f"  • Macro-F1       : {metrics.get('macro_f1', metrics.get('Macro-F1', 0.0)):.4f}")
    print(f"  • Head-Micro-F1  : {metrics.get('head_micro_f1', metrics.get('Head-Micro-F1', 0.0)):.4f}")
    print(f"  • Head-Macro-F1  : {metrics.get('Head-Macro-F1', metrics.get('head_macro_f1', 0.0)):.4f}")
    print(f"  • Tail-Micro-F1  : {metrics.get('tail_micro_f1', metrics.get('Tail-Micro-F1', 0.0)):.4f}")
    print(f"  • Tail-Macro-F1  : {metrics.get('tail_macro_f1', metrics.get('Tail-Macro-F1', 0.0)):.4f}")
    print("="*60)
    
    return metrics, y_pred


def evaluate_llama3_few_shot(
    test_dataset,
    train_dataset,
    candidate_labels: List[str],
    head_indices: list,
    tail_indices: list,
    dataset_name: str = "UK-LEX-18",
    n_shots: int = 3,
    max_samples: Optional[int] = None,
    checkpoint_dir: str = "results",
    save_every: int = 100,
    batch_size: int = 32,
    max_new_tokens: int = 100,
    llama_model = None,
    tokenizer = None
):
    """
    Executes Advanced Few-Shot Llama-3 evaluation with in-context legal exemplars,
    official chat markup formatting, expanded token budget (100 tokens), and partial checkpointing.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    clean_name = dataset_name.lower().replace("-", "")
    ckpt_path = os.path.join(checkpoint_dir, f"llama3_preds_fewshot_{clean_name}_ckpt.npy")

    prompt_builder = LlamaPromptBuilder(candidate_labels=candidate_labels, domain_name=dataset_name)
    
    # Sample n_shots diverse exemplars from historical train_dataset
    exemplar_docs = []
    if train_dataset is not None and len(train_dataset) > 0:
        step = max(1, len(train_dataset) // n_shots)
        for i in range(0, len(train_dataset), step):
            if len(exemplar_docs) < n_shots:
                exemplar_docs.append(train_dataset[i])

    print(f"\n Building Few-Shot Prompts ({n_shots}-shot) with chat templates for {dataset_name}...")
    prompts = []
    for sample in tqdm(test_dataset, desc="Generating Few-Shot Prompts"):
        text = sample.get('body', sample.get('text', ''))
        title = sample.get('title', '')
        prompt = prompt_builder.build_few_shot_prompt(
            document_text=text,
            exemplar_docs=exemplar_docs,
            document_title=title,
            tokenizer=tokenizer,
            text_length=3500
        )
        prompts.append(prompt)

    if max_samples is not None:
        prompts = prompts[:max_samples]
        if hasattr(test_dataset, 'select'):
            test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))

    # Check for existing checkpoint
    all_pred_vectors = []
    start_idx = 0
    if os.path.exists(ckpt_path):
        try:
            saved_preds = np.load(ckpt_path)
            all_pred_vectors = list(saved_preds)
            start_idx = len(all_pred_vectors)
            print(f" Resuming {dataset_name} Few-Shot evaluation from checkpoint at doc {start_idx:,}/{len(prompts):,}...")
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {ckpt_path} ({e}). Starting fresh.")
            all_pred_vectors = []
            start_idx = 0

    if start_idx < len(prompts):
        # Check model and tokenizer
        if llama_model is None and 'llama_model' in globals():
            llama_model = globals()['llama_model']
        if tokenizer is None and 'tokenizer' in globals():
            tokenizer = globals()['tokenizer']

        if llama_model is not None and tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            print(f" Running Batched PyTorch CUDA Few-Shot Inference (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
            pbar = tqdm(total=len(prompts), initial=start_idx, desc=f"Llama-3 Few-Shot ({dataset_name})")

            for b_start in range(start_idx, len(prompts), batch_size):
                b_end = min(b_start + batch_size, len(prompts))
                batch_prompts = prompts[b_start:b_end]

                inputs = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096
                ).to("cuda")

                with torch.no_grad():
                    outputs = llama_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id
                    )

                input_len = inputs.input_ids.shape[1]
                for seq in outputs:
                    gen_tokens = seq[input_len:]
                    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    pred_vector = parse_llm_json_output(gen_text, candidate_labels)
                    all_pred_vectors.append(pred_vector)

                del inputs, outputs
                torch.cuda.empty_cache()
                pbar.update(b_end - b_start)

                if len(all_pred_vectors) % save_every < batch_size or b_end == len(prompts):
                    np.save(ckpt_path, np.array(all_pred_vectors, dtype=np.float32))
            pbar.close()

    y_pred = np.array(all_pred_vectors, dtype=np.float32)
    y_true = test_dataset['labels']
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    else:
        y_true = np.array(y_true, dtype=np.float32)
    y_true = y_true[:len(y_pred)]

    metrics = compute_temporal_f1_metrics(y_true, y_pred, head_indices, tail_indices)

    print("\n" + "="*60)
    print(f"{dataset_name} FEW-SHOT LLAMA-3 TEST RESULTS")
    print("="*60)
    print(f"  • Micro-F1       : {metrics.get('micro_f1', 0.0):.4f}")
    print(f"  • Macro-F1       : {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  • Head-Micro-F1  : {metrics.get('head_micro_f1', 0.0):.4f}")
    print(f"  • Head-Macro-F1  : {metrics.get('head_macro_f1', 0.0):.4f}")
    print(f"  • Tail-Micro-F1  : {metrics.get('tail_micro_f1', 0.0):.4f}")
    print(f"  • Tail-Macro-F1  : {metrics.get('tail_macro_f1', 0.0):.4f}")
    print("="*60)

    return metrics, y_pred


class LlamaZeroShotEvaluator:
    """
    Evaluator wrapper for running zero-shot Llama-3 inference across chronological test splits.
    """
    def __init__(self, prompt_builder: LlamaPromptBuilder, model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.prompt_builder = prompt_builder
        self.model_id = model_id

    def generate_prompt_batch(self, dataset) -> List[str]:
        """
        Generates prompts for an entire test split.
        """
        prompts = []
        for sample in dataset:
            text = sample.get('text', sample.get('body', ''))
            title = sample.get('title', '')
            prompt = self.prompt_builder.build_zero_shot_prompt(text, title)
            prompts.append(prompt)
        return prompts
