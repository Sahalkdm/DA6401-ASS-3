"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """

    dk = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1) / math.sqrt(dk))

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
 
    # Causal mask
    causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),diagonal=1).unsqueeze(0)
 
    # broadcast over seq_q dimension
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
 
    tgt_mask = causal_mask | pad_mask  # (batch, 1, tgt_len, tgt_len)
 
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # depth per head
        
        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
 
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split last dim into (num_heads, d_k) and transpose for attention."""
        batch, seq_len, _ = x.size()
        return x.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch = query.size(0)
 
        # Linear projections + split into heads
        Q = self._split_heads(self.W_q(query))   
        K = self._split_heads(self.W_k(key))     
        V = self._split_heads(self.W_v(value))   
 
        # Scaled dot-product attention
        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
 
        # dropout on attention weights (for regularisation)
        attn_out = self.dropout(attn_out)
 
        # Concatenate heads: (batch, seq_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
 
        output = self.W_o(attn_out)
 
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute PE table
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
 
        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions
 
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # not a parameter, but moves with .to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # self-attention + Add & Norm
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
 
        # FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
 
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
 
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
 
        # Sub-layer 2
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
 
        # Sub-layer 3: FFN + Add & Norm
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
 
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    GDRIVE_FILE_ID = "YOUR_GDRIVE_FILE_ID_HERE"
    CHECKPOINT_FILENAME = "best_model.pt"

    def __init__(
        self,
        src_vocab_size: int=None,
        tgt_vocab_size: int=None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,

        checkpoint_path=None,
        download_if_missing=True,
        gdrive_file_id=None,

        src_vocab=None,
        tgt_vocab=None,

        pad_idx:   int   = 1,
        sos_idx:   int   = 2,
        eos_idx:   int   = 3,
    ) -> None:
        super().__init__()

        import os
        import spacy

        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx

        ckpt = None

        # -------------------------------------------------
        # OPTIONAL DOWNLOAD
        # -------------------------------------------------

        if checkpoint_path is not None:

            if (download_if_missing and not os.path.exists(checkpoint_path)):

                if gdrive_file_id is None:
                    raise ValueError("gdrive_file_id required.")

                self._download_weights(gdrive_file_id, checkpoint_path)

            # Load checkpoint if available
            if os.path.exists(checkpoint_path):

                print(f"Loading checkpoint: {checkpoint_path}")

                ckpt = torch.load(checkpoint_path, map_location="cpu")

                cfg = ckpt.get("model_config", {})

                d_model = cfg.get("d_model", d_model)
                N = cfg.get("N", N)
                num_heads = cfg.get("num_heads", num_heads)
                d_ff = cfg.get("d_ff", d_ff)
                dropout = cfg.get("dropout", dropout)

                src_vocab = ckpt.get("src_vocab")
                tgt_vocab = ckpt.get("tgt_vocab")

                if src_vocab is None or tgt_vocab is None:
                    raise RuntimeError("Checkpoint missing vocabularies.")

                src_vocab_size = len(src_vocab)
                tgt_vocab_size = len(tgt_vocab)


        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError("Vocabulary sizes must be provided.")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if tgt_vocab is not None:
            self.tgt_itos = {v: k for k, v in tgt_vocab.items()}

        try:
            self._nlp_de = spacy.load("de_core_news_sm")

        except OSError:

            print("Downloading de_core_news_sm ...")

            from spacy.cli import download as spacy_download

            spacy_download("de_core_news_sm")

            self._nlp_de = spacy.load("de_core_news_sm")

        self.d_model = d_model

        self.src_embed = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.tgt_embed = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=pad_idx
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            dropout
        )

        enc_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(enc_layer, N)

        dec_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.decoder = Decoder(dec_layer, N)

        self.output_proj = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self._init_weights()

        if ckpt is not None:

            self.load_state_dict(ckpt["model_state_dict"])

            print("Checkpoint weights loaded.")

        # super().__init__()
        # self.d_model = d_model
        # self.pad_idx = pad_idx
        # self.sos_idx = sos_idx
        # self.eos_idx = eos_idx

        # self.src_vocab = src_vocab
        # self.tgt_vocab = tgt_vocab
        
        # # TODO: Instantiate 
        # # init should also load the model weights if checkpoint path provided, download the .pth file like this

        # ckpt_path = self._download_weights()
        # ckpt = torch.load(ckpt_path, map_location="cpu")

        # # Architecture config saved during training
        # cfg = ckpt.get("model_config", {})
        # d_model = cfg.get("d_model", d_model)
        # N = cfg.get("N", N)
        # num_heads = cfg.get("num_heads", num_heads)
        # d_ff = cfg.get("d_ff", d_ff)
        # dropout = cfg.get("dropout", dropout)

        # src_stoi = ckpt.get("src_vocab")   # dict
        # tgt_stoi = ckpt.get("tgt_vocab")   # dict

        # if src_stoi is None or tgt_stoi is None:
        #     raise RuntimeError(
        #         "Checkpoint does not contain src_vocab / tgt_vocab. "
        #         "Re-save your checkpoint using the updated save_checkpoint() in train.py."
        #     )
        
        # self.src_vocab = src_stoi   # word -> int
        # self.tgt_vocab = tgt_stoi   # word -> int
        # self.tgt_itos = {v: k for k, v in tgt_stoi.items()}  # int -> word
 
        # src_vocab_size = len(src_stoi)
        # tgt_vocab_size = len(tgt_stoi)

        # # Load spaCy German tokeniser
        # import spacy
        # try:
        #     self._nlp_de = spacy.load("de_core_news_sm")
        # except OSError:
        #     from spacy.cli import download as spacy_dl
        #     spacy_dl("de_core_news_sm")
        #     self._nlp_de = spacy.load("de_core_news_sm")

        # self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        # self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        # self.pos_enc = PositionalEncoding(d_model, dropout)
 
        # enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        # self.encoder = Encoder(enc_layer, N)
 
        # dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        # self.decoder = Decoder(dec_layer, N)
 
        # self.output_proj = nn.Linear(d_model, tgt_vocab_size)
 
        # self._init_weights()
 
        # # ── Step 5: Load trained weights ──────────────────────────────
        # self.load_state_dict(ckpt["model_state_dict"])
        # print("Transformer loaded successfully.")

        # Embeddings
        # self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        # self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
 
        # # Positional encoding
        # self.pos_enc = PositionalEncoding(d_model, dropout)
 
        # # Encoder stack
        # enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        # self.encoder = Encoder(enc_layer, N)
 
        # # Decoder stack
        # dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        # self.decoder = Decoder(dec_layer, N)
 
        # # Final linear projection to vocab
        # self.output_proj = nn.Linear(d_model, tgt_vocab_size)
 
        # # Weight tying
        # # self.output_proj.weight = self.tgt_embed.weight
 
        # # Kaiming / Xavier init
        # self._init_weights()
 
        # # Load checkpoint if path provided
        # if checkpoint_path is not None:
        #     self.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['model_state_dict'])

        # if checkpoint_path is not None:
        #     gdown.download(id="<.pth drive id>", output=checkpoint_path, quiet=False)
    
    def _download_weights(self, gdrive_file_id, checkpoint_path) -> str:
        """
        Download checkpoint from Google Drive using gdown.
        Returns the local path to the downloaded file.
        [AI Generated]
        """
        import os
        ckpt_path = checkpoint_path
 
        if not os.path.exists(ckpt_path):
            try:
                import gdown
            except ImportError:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
                import gdown
 
            url = f"https://drive.google.com/uc?id={gdrive_file_id}"
            print(f"Downloading weights from Google Drive …")
            gdown.download(url, ckpt_path, quiet=False)
 
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Failed to download checkpoint. "
                f"Check that GDRIVE_FILE_ID='{gdrive_file_id}' is correct "
                f"and the file is shared publicly ('Anyone with the link')."
            )
 
        return ckpt_path
    
    def _init_weights(self):
        """Initialize parameters using Xavier uniform (as in the original paper)."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        dec_out = self.decoder(tgt_emb, memory, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        """
        End-to-end German → English translation.
 
        Args:
            src_sentence : Raw German string.
            max_len      : Maximum tokens to generate.
 
        Returns:
            Translated English string.
        """
        self.eval()
        device = next(self.parameters()).device
 
        # Tokenise German with spaCy
        tokens = [tok.text.lower() for tok in self._nlp_de(src_sentence)]
 
        # Numericalize
        unk = self.src_vocab.get("<unk>", 0)
        src_ids = ([self.src_vocab.get("<sos>", self.sos_idx)] + [self.src_vocab.get(t, unk) for t in tokens]+ [self.src_vocab.get("<eos>", self.eos_idx)])
        src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, self.pad_idx).to(device)
 
        # Greedy autoregressive decoding
        with torch.no_grad():
            memory = self.encode(src, src_mask)
            ys = torch.tensor([[self.sos_idx]], dtype=torch.long, device=device)
 
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.pad_idx).to(device)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_token], dim=1)
                if next_token.item() == self.eos_idx:
                    break
 
        # Detokenize
        special = {"<sos>", "<eos>", "<pad>", "<unk>"}
        out_tokens = []
        for idx in ys[0, 1:].tolist():
            if idx == self.eos_idx:
                break
            tok = self.tgt_itos.get(idx, "<unk>")
            if tok not in special:
                out_tokens.append(tok)
 
        return " ".join(out_tokens)
    
    # def infer(self, src_sentence: str, device: str = "cpu", max_len: int = 100) -> str:
    #     """
    #     Translates a German sentence to English using greedy autoregressive decoding.
        
    #     Args:
    #         src_sentence: The raw German text.
            
            
    #     Returns:
    #         The fully translated English string, detokenized and clean.
    #     """
    #     import spacy
    #     self.eval()
    #     nlp_de = spacy.load("de_core_news_sm")
 
    #     # Tokenise and numericalize
    #     tokens = [tok.text.lower() for tok in nlp_de(src_sentence)]
    #     src_indices = (
    #         [self.src_vocab.get('<sos>', 2)]
    #         + [self.src_vocab.get(t, self.src_vocab.get('<unk>', 0)) for t in tokens]
    #         + [self.src_vocab.get('<eos>', 3)]
    #     )
    #     src = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    #     src_mask = make_src_mask(src, self.pad_idx).to(device)
 
    #     with torch.no_grad():
    #         memory = self.encode(src, src_mask)
    #         ys = torch.tensor([[self.sos_idx]], dtype=torch.long).to(device)
 
    #         for _ in range(max_len):
    #             tgt_mask = make_tgt_mask(ys, self.pad_idx).to(device)
    #             logits = self.decode(memory, src_mask, ys, tgt_mask)
    #             next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #             ys = torch.cat([ys, next_token], dim=1)
    #             if next_token.item() == self.eos_idx:
    #                 break
 
    #     # Decode indices to tokens
    #     itos = {v: k for k, v in self.tgt_vocab.items()}
    #     output_tokens = []
    #     for idx in ys[0, 1:].tolist():
    #         if idx == self.eos_idx:
    #             break
    #         tok = itos.get(idx, '<unk>')
    #         if tok not in ('<sos>', '<eos>', '<pad>'):
    #             output_tokens.append(tok)
 
    #     return " ".join(output_tokens)
 