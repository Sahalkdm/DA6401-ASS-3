"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask

from dataset_copy import PAD_IDX, SOS_IDX, EOS_IDX, build_datasets, build_dataloaders
from lr_scheduler import NoamScheduler

import wandb
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)

            smooth_dist[:, self.pad_idx] = 0.0
            # Mask entire rows where target is <pad>
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average over non-pad tokens
        non_pad = (~pad_mask).sum().item()
        if non_pad == 0:
            return loss.sum() * 0.0
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()

    total_loss = 0
    total_steps = 0

    phase = "train" if is_train else "val"
    pbar = tqdm(data_iter, desc=f"Epoch {epoch_num} [{phase}]", leave=False)

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX).to(device)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten for loss
            batch_size, tgt_seq_len, vocab_size = logits.shape
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            targets_flat = tgt_labels.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            total_loss  += loss.item()
            total_steps += 1
 
            pbar.set_postfix(loss=f"{loss.item():.4f}")
 
            # Log step-level metrics to W&B during training
            if is_train and wandb.run is not None:
                current_lr = optimizer.param_groups[0]['lr'] if optimizer else 0.0
                wandb.log({
                    "train/step_loss": loss.item(),
                    "train/lr": current_lr,
                    "train/step": epoch_num * len(data_iter) + batch_idx,
                })
 
    avg_loss = total_loss / max(total_steps, 1)
    
    # Log epoch-level metrics to W&B
    if wandb.run is not None:
        wandb.log({
            f"{phase}/epoch_loss": avg_loss,
            "epoch": epoch_num,
        })
 
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding

    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX).to(device)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
 
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    
    model.eval()
    bleu_metric = BLEU(effective_order=True)
 
    hypotheses  = []
    references  = []
 
    # Special token strings to strip from output
    special = {'<sos>', '<eos>', '<pad>', '<unk>'}
 
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
 
            src_mask = make_src_mask(src, PAD_IDX).to(device)
 
            output = greedy_decode(
                model, src, src_mask, max_len,
                start_symbol=SOS_IDX,
                end_symbol=EOS_IDX,
                device=device,
            )  # (1, out_len)
 
            # Decode hypothesis
            hyp_tokens = []
            for idx in output[0, 1:].tolist():   # skip <sos>
                if idx == EOS_IDX:
                    break
                tok = tgt_vocab.lookup_token(idx)
                if tok not in special:
                    hyp_tokens.append(tok)
 
            # Decode reference 
            ref_tokens = []
            for idx in tgt[0, 1:].tolist():
                if idx == EOS_IDX:
                    break
                tok = tgt_vocab.lookup_token(idx)
                if tok not in special:
                    ref_tokens.append(tok)
 
            hypotheses.append(" ".join(hyp_tokens))
            references.append(" ".join(ref_tokens))
 
    result = bleu_metric.corpus_score(hypotheses, [references])
    return result.score


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    model_config = {
        'src_vocab_size': model.src_embed.num_embeddings,
        'tgt_vocab_size': model.tgt_embed.num_embeddings,
        'd_model': model.d_model,
        'N': len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads,
        'd_ff': model.encoder.layers[0].ffn.linear1.out_features,
        'dropout': model.encoder.layers[0].dropout.p,
    }
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': model_config,
        # Vocab dicts needed by Transformer.__init__() for inference
        'src_vocab': model.src_vocab if isinstance(model.src_vocab, dict) else model.src_vocab.stoi,
        'tgt_vocab': model.tgt_vocab if isinstance(model.tgt_vocab, dict) else model.tgt_vocab.stoi,
    }, path)
    print(f"Checkpoint saved → {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    # device = next(model.parameters()).device
    checkpoint = torch.load(path, map_location='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded ← {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    config = dict(
        # Model
        d_model = 256,
        N = 4, # encoder/decoder layers
        num_heads = 8,
        d_ff = 512,
        dropout = 0.1,
        # Training
        batch_size = 128,
        num_epochs = 25,
        warmup_steps = 4000,
        label_smoothing = 0.1,
        # Data
        min_freq = 2,
        max_src_len = 100,
        max_tgt_len = 100,

        seed = 42,
        checkpoint_dir = "checkpoints",
    )

    wandb.init(
        project = "da6401-a3",
        name = f"transformer_d{config['d_model']}_N{config['N']}_h{config['num_heads']}",
        config = config,
        tags = ["transformer", "de-en", "multi30k"],
    )
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    print("Loading Multi30k dataset…")
    train_ds, val_ds, test_ds = build_datasets(
        min_freq    = cfg.min_freq,
        max_src_len = cfg.max_src_len,
        max_tgt_len = cfg.max_tgt_len,
    )

    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    print(f"Source vocab size : {len(src_vocab)}")
    print(f"Target vocab size : {len(tgt_vocab)}")
    print(f"Train examples : {len(train_ds)}")
    print(f"Val   examples : {len(val_ds)}")
    print(f"Test  examples : {len(test_ds)}")

    wandb.config.update({
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
    })

    # data loaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size = cfg.batch_size,
    )

    # model
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model = cfg.d_model,
        N = cfg.N,
        num_heads = cfg.num_heads,
        d_ff = cfg.d_ff,
        dropout = cfg.dropout,
        src_vocab=src_vocab.stoi,
        tgt_vocab=tgt_vocab.stoi,
        pad_idx = PAD_IDX,
        sos_idx = SOS_IDX,
        eos_idx = EOS_IDX,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters  : {total_params:,}")
    wandb.config.update({"total_params": total_params})

    wandb.watch(model, log="all", log_freq=100)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 1.0,    # base LR = 1; Noam scheduler scales it
        betas = (0.9, 0.98),
        eps = 1e-9,
    )

    # Scheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model = cfg.d_model,
        warmup_steps = cfg.warmup_steps,
    )

    # loss fn
    loss_fn = LabelSmoothingLoss(
        vocab_size = len(tgt_vocab),
        pad_idx = PAD_IDX,
        smoothing  = cfg.label_smoothing,
    )

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}")
 
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num = epoch,
            is_train = True,
            device = device,
        )
 
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            optimizer = None,
            scheduler = None,
            epoch_num = epoch,
            is_train = False,
            device = device,
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss : {val_loss:.4f}")
        print(f"  LR : {current_lr:.6f}")
 
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/ppl": math.exp(train_loss),
            "val/ppl": math.exp(val_loss),
            "lr": current_lr,
        })
 
        # Save best model
        ckpt_path = os.path.join(cfg.checkpoint_dir, f"checkpoint_epoch{epoch+1}.pt")
        save_checkpoint(model, optimizer, scheduler, epoch + 1, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            save_checkpoint(model, optimizer, scheduler, epoch + 1, best_path)
            print(f" -> New best model saved (val_loss={val_loss:.4f})")
            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch + 1
 
        # Log a few example translations every 5 epochs
        if (epoch + 1) % 5 == 0:
            _log_translation_examples(model, val_ds, tgt_vocab, device)

    # Calculate bleu on test set
    print("Evaluating BLEU on test set…")
    # Load best checkpoint for evaluation
    best_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    if os.path.exists(best_path):
        load_checkpoint(best_path, model)

    model = Transformer(
        checkpoint_path="best_model.pt",
        download_if_missing=True,
        gdrive_file_id="19u5b6lNS17qGB5j5zbxhOoHLyYkt3Ets",
    )
 
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    print(f"  Test BLEU : {bleu:.2f}")
 
    wandb.log({"test/bleu": bleu})
    wandb.run.summary["test_bleu"] = bleu
 
    wandb.finish()
    print("Training complete.")

# ══════════════════════════════════════════════════════════════════════
#  LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════
 
def _log_translation_examples(model, val_ds, tgt_vocab, device, n=3):
    """
    Log a few German→English translation examples to W&B as a Table.
    """
    import random
    from dataset import PAD_IDX
 
    model.eval()
    examples = random.sample(range(len(val_ds)), min(n, len(val_ds)))
    special  = {'<sos>', '<eos>', '<pad>', '<unk>'}
 
    rows = []
    src_vocab_itos = {v: k for k, v in val_ds.src_vocab.stoi.items()}
 
    for idx in examples:
        src_ids, tgt_ids = val_ds[idx]
        src = src_ids.unsqueeze(0).to(device)
        src_mask = make_src_mask(src, PAD_IDX).to(device)
 
        output = greedy_decode(
            model, src, src_mask, max_len=100,
            start_symbol=SOS_IDX, end_symbol=EOS_IDX, device=device
        )
 
        # Source German
        src_words = [src_vocab_itos.get(i.item(), '<unk>') for i in src_ids
                     if src_vocab_itos.get(i.item(), '') not in special]
 
        # Reference English
        ref_words = [tgt_vocab.lookup_token(i.item()) for i in tgt_ids
                     if tgt_vocab.lookup_token(i.item()) not in special]
 
        # Hypothesis
        hyp_words = []
        for i in output[0, 1:].tolist():
            if i == EOS_IDX:
                break
            tok = tgt_vocab.lookup_token(i)
            if tok not in special:
                hyp_words.append(tok)
 
        rows.append([
            " ".join(src_words),
            " ".join(ref_words),
            " ".join(hyp_words),
        ])
 
    table = wandb.Table(columns=["German (src)", "English (ref)", "Translation (hyp)"], data=rows)
    wandb.log({"translation_examples": table})

if __name__ == "__main__":
    run_training_experiment()