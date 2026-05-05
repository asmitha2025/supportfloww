"""
SupportMind Ultimate — DeBERTa-v3-xsmall Fine-Tuning (CPU)
Custom training loop — fully unfreezes model for maximum accuracy.

RTX 4050 has CUDA 12.9 driver / PyTorch 12.1 mismatch, so we train on CPU.
DeBERTa-v3-xsmall (70M params) fits comfortably in 16 GB system RAM.
"""
import os
import sys
import gc
import json
import time

os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import logging
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.utils.class_weight import compute_class_weight
import psutil

# ── Configuration ──────────────────────────────────────────────────────────
MODEL_NAME = "microsoft/deberta-v3-xsmall"
MAX_LENGTH = 128
BATCH_SIZE = 4           # Larger batch on CPU is fine (enough RAM)
GRADIENT_ACC = 4         # Effective batch = 16
EPOCHS = 8
LR = 2e-5               # Standard fine-tuning LR for transformers
WARMUP_STEPS = 50        # Linear warmup
EVAL_EVERY = 25          # Evaluate more frequently to catch improvements
PATIENCE = 5             # Early stopping patience (in eval rounds)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'deberta_ultimate')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

process = psutil.Process(os.getpid())


def log_memory(tag=""):
    ram = process.memory_info().rss / 1024**2
    logger.info(f"[MEM {tag}] RAM={ram:.0f}MB")


def tokenize_dataframe(df, tokenizer):
    """Tokenize a DataFrame and return a TensorDataset."""
    encodings = tokenizer(
        df['text'].tolist(),
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    return TensorDataset(encodings['input_ids'], encodings['attention_mask'], labels)


@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device):
    """Evaluate model on validation set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for ids, mask, labels in val_loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        outputs = model(input_ids=ids, attention_mask=mask)
        total_loss += loss_fn(outputs.logits, labels).item()
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    model.train()
    return correct / total, total_loss / len(val_loader)


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Simple linear warmup then linear decay scheduler."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = 'cpu'

    logger.info("=" * 60)
    logger.info("SupportMind Ultimate — DeBERTa-v3-xsmall (Full Fine-Tune)")
    logger.info(f"  Device: CPU | LR: {LR} | Epochs: {EPOCHS}")
    logger.info(f"  Batch: {BATCH_SIZE} x {GRADIENT_ACC} = {BATCH_SIZE*GRADIENT_ACC} effective")
    logger.info("=" * 60)

    # ── 1. Load tokenizer ─────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── 2. Load and tokenize data ─────────────────────────────────────────
    logger.info("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    logger.info(f"Train: {len(train_df)} | Val: {len(val_df)}")

    # Compute class weights for balanced loss
    labels_arr = train_df['label'].values
    unique_l = sorted(set(labels_arr))
    cw = compute_class_weight('balanced', classes=np.array(unique_l), y=labels_arr)
    class_weights = torch.tensor(
        [cw[i] if i < len(cw) else 1.0 for i in range(8)], dtype=torch.float32
    ).to(device)
    logger.info(f"Class weights: {[round(w, 3) for w in class_weights.tolist()]}")

    train_dataset = tokenize_dataframe(train_df, tokenizer)
    val_dataset = tokenize_dataframe(val_df, tokenizer)
    del train_df, val_df
    gc.collect()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── 3. Load model — FULL fine-tuning ──────────────────────────────────
    logger.info(f"Loading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)

    # NO freezing — full fine-tune for maximum accuracy
    # DeBERTa-v3-xsmall is only 70M params, fits in 16GB RAM with AdamW
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"Params: {trainable:,} / {total_p:,} ({trainable/total_p*100:.1f}% trainable)")

    model.to(device)
    model.train()
    log_memory("after model.to(device)")

    # ── 4. Optimizer + Scheduler ──────────────────────────────────────────
    # Differential learning rates: lower LR for backbone, higher for head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'pooler' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR},
        {'params': head_params, 'lr': LR * 10},  # 10x LR for randomly init head
    ], weight_decay=0.01)

    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    total_optimizer_steps = (len(train_loader) // GRADIENT_ACC) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, WARMUP_STEPS, total_optimizer_steps)

    logger.info(f"Total optimizer steps: {total_optimizer_steps}")
    logger.info(f"Warmup steps: {WARMUP_STEPS}")
    log_memory("after optimizer init")

    # ── 5. Training loop ──────────────────────────────────────────────────
    logger.info("Starting training...")
    sys.stdout.flush()

    best_acc = 0.0
    best_loss = float('inf')
    best_epoch = -1
    global_step = 0
    no_improve_count = 0

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (ids, mask, labels) in enumerate(train_loader):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)

            # Forward + backward
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(outputs.logits, labels) / GRADIENT_ACC
            loss.backward()

            epoch_loss += loss.item() * GRADIENT_ACC
            preds = outputs.logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += labels.size(0)

            del ids, mask, labels, outputs, preds, loss

            # Optimizer step
            if (batch_idx + 1) % GRADIENT_ACC == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0:
                    elapsed = time.time() - t0
                    pct = (global_step / total_optimizer_steps) * 100
                    avg_loss = epoch_loss / (batch_idx + 1)
                    train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
                    cur_lr = scheduler.get_last_lr()[0]
                    logger.info(
                        f"[{pct:5.1f}%] Epoch {epoch+1}/{EPOCHS} Step {global_step}/{total_optimizer_steps} "
                        f"| loss={avg_loss:.4f} acc={train_acc:.3f} | lr={cur_lr:.2e} | {elapsed:.0f}s"
                    )
                    sys.stdout.flush()

                # Periodic evaluation
                if global_step % EVAL_EVERY == 0:
                    val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)
                    logger.info(
                        f"  >> EVAL step {global_step}: val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
                    )
                    sys.stdout.flush()

                    if val_acc > best_acc:
                        best_acc = val_acc
                        best_loss = val_loss
                        best_epoch = epoch + 1
                        no_improve_count = 0
                        logger.info(f"  >> New best! Saving model (acc={best_acc:.4f})")
                        model.save_pretrained(OUTPUT_DIR)
                        tokenizer.save_pretrained(OUTPUT_DIR)
                    else:
                        no_improve_count += 1
                        if no_improve_count >= PATIENCE:
                            logger.info(f"  >> Early stopping (no improvement for {PATIENCE} evals)")
                            break

        if no_improve_count >= PATIENCE:
            break

        # End of epoch eval
        val_acc, val_loss = evaluate(model, val_loader, loss_fn, device)
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} done ({elapsed:.0f}s) | "
            f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f} val_loss={val_loss:.4f}"
        )
        sys.stdout.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            logger.info(f"  >> New best! Saving model (acc={best_acc:.4f})")
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    # ── 6. Save final results ─────────────────────────────────────────────
    results = {
        'best_val_accuracy': best_acc,
        'best_val_loss': best_loss,
        'best_epoch': best_epoch,
        'model_name': MODEL_NAME,
        'max_length': MAX_LENGTH,
        'epochs_trained': min(epoch + 1, EPOCHS),
        'effective_batch_size': BATCH_SIZE * GRADIENT_ACC,
        'learning_rate': LR,
    }
    with open(os.path.join(RESULTS_DIR, 'ultimate_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Training complete! Best val accuracy: {best_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
