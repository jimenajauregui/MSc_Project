import torch
import torch.nn as nn
from transformers import AutoModel


class LegalBertClassifier(nn.Module):
    """
    Legal-BERT Multi-Label Classifier Architecture.
    
    Extracts the [CLS] token representation from Legal-BERT (`nlpaueb/legal-bert-base-uncased`)
    and applies a linear classification head outputting raw logits for K multi-hot targets.
    
    Ref: Chalkidis et al. (2020), Chalkidis & Søgaard (2022)
    """
    def __init__(
        self, 
        model_name: str = "nlpaueb/legal-bert-base-uncased", 
        num_labels: int = 18, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract leading [CLS] token contextualized embedding
        cls_output = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(cls_output)
        logits = self.classifier(pooled)
        return logits


class LegalBertLWAN(nn.Module):
    """
    Legal-BERT Label-Wise Attention Network (BERT-LWAN) Architecture.
    
    Unlike standard BERT which relies solely on the [CLS] token, BERT-LWAN employs a per-label 
    attention mechanism across all sequence token representations H in the document. This allows
    the model to focus on distinct text segments relevant to each specific category k.
    
    Ref: Chalkidis et al. (2020) "LEGAL-BERT: The Muppets straight out of Law School"
    Ref: Chalkidis & Søgaard (2022) "Empirical Evaluation of Multi-Label Classification under Drift"
    """
    def __init__(
        self,
        model_name: str = "nlpaueb/legal-bert-base-uncased",
        num_labels: int = 18,
        attention_dim: int = 200,
        dropout: float = 0.1
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.attention_dim = attention_dim
        
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Per-label attention projection: H (B, N, hidden_size) -> Q (B, N, attention_dim)
        self.attention_proj = nn.Linear(hidden_size, attention_dim)
        # Query parameters U for K labels: (num_labels, attention_dim)
        self.label_queries = nn.Parameter(torch.Tensor(num_labels, attention_dim))
        nn.init.xavier_uniform_(self.label_queries)
        
        self.dropout = nn.Dropout(dropout)
        
        # Per-label classification weights: W_out (num_labels, hidden_size) & b_out (num_labels)
        self.classifier_weight = nn.Parameter(torch.Tensor(num_labels, hidden_size))
        self.classifier_bias = nn.Parameter(torch.Tensor(num_labels))
        nn.init.xavier_uniform_(self.classifier_weight)
        nn.init.zeros_(self.classifier_bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Extract full sequence contextualized token embeddings H: (B, N, hidden_size)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, N, hidden_size)
        
        # 1. Project token embeddings: Q = tanh(H @ W_a) -> (B, N, attention_dim)
        Q = torch.tanh(self.attention_proj(H))
        
        # 2. Compute per-label attention scores: S = Q @ U^T -> (B, N, K)
        scores = torch.matmul(Q, self.label_queries.T)  # (B, N, num_labels)
        
        # 3. Apply attention_mask: mask padding tokens (-1e9) before softmax
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(mask_expanded == 0, -1e9)
        
        # 4. Compute softmax weights over sequence dimension N -> (B, K, N)
        attn_weights = torch.softmax(scores, dim=1).transpose(1, 2)  # (B, K, N)
        
        # 5. Compute label-specific context representations: D = attn_weights @ H -> (B, K, hidden_size)
        D = torch.matmul(attn_weights, H)  # (B, K, hidden_size)
        D = self.dropout(D)
        
        # 6. Compute per-label raw logits: z_k = (D_k . w_k) + b_k
        logits = (D * self.classifier_weight.unsqueeze(0)).sum(dim=-1) + self.classifier_bias
        return logits

