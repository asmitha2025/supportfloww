# src/train_router.py
# Fine-tune DistilBERT for 8-class ticket routing
# SupportMind v1.0 — Asmitha
#
# Memory-optimized for machines with limited RAM:
#   - max_length=64 (tickets are short, saves ~4x memory vs 256)
#   - batch_size=2 (minimal footprint)
#   - gradient_accumulation_steps=8 (effective batch=16)
#   - fp16=True if CUDA available
#   - Datasets cleared before model loading

import os
import sys
import gc

# Disable TensorFlow to prevent DLL loading errors under Application Control policies
os.environ['USE_TF'] = '0'
os.environ['USE_JAX'] = '0'
# Limit torch threads to reduce memory pressure
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
import torch
import logging
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    TrainerCallback
)
from transformers.trainer_utils import get_last_checkpoint
import psutil
from datasets import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ticket_classifier')

# Shorter max_length — support tickets are typically short
# 64 tokens is enough to capture intent from these tickets
MAX_LENGTH = 64


class MemoryProfilerCallback(TrainerCallback):
    """Logs memory usage + progress summary every N steps."""
    def __init__(self, total_steps: int):
        import os
        self.process = psutil.Process(os.getpid())
        self.total_steps = total_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            mem_mb = self.process.memory_info().rss / (1024 * 1024)
            pct = (state.global_step / self.total_steps) * 100 if self.total_steps else 0
            logger.info(
                f"[{pct:5.1f}%] Step {state.global_step}/{self.total_steps} "
                f"| Epoch {state.epoch:.2f} | RAM: {mem_mb:.0f} MB"
            )


def compute_metrics(eval_pred):
    """Compute accuracy metric for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean()
    return {"accuracy": float(accuracy)}


def main():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    val_path = os.path.join(DATA_DIR, 'val.csv')
    
    if not os.path.exists(train_path):
        logger.error(f"Training data not found at {train_path}. Run data/preprocess.py first.")
        sys.exit(1)

    # ── Step 1: Load & tokenize data ──────────────────────
    logger.info("Loading processed datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    logger.info(f"Train: {len(train_df)} samples, Val: {len(val_df)} samples")
    logger.info(f"Label distribution:\n{train_df['label'].value_counts().to_string()}")

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = device == "cuda"
    logger.info(f"Device: {device} | FP16: {use_fp16}")

    # Convert to HF Datasets  
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

    # Free DataFrame memory before tokenization
    del train_df, val_df
    gc.collect()

    logger.info("Initializing Tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)

    logger.info("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=64)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, batch_size=64)

    # Free raw datasets
    del train_dataset, val_dataset
    gc.collect()

    # ── Step 2: Compute class weights for imbalanced data ─
    from sklearn.utils.class_weight import compute_class_weight
    
    labels_array = tokenized_train['label']
    unique_labels = sorted(set(labels_array))
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array(unique_labels),
        y=np.array(labels_array)
    )
    # Map to all 8 classes (some might be missing)
    weight_dict = {c: w for c, w in zip(unique_labels, class_weights)}
    weights_tensor = torch.tensor(
        [weight_dict.get(i, 1.0) for i in range(8)], dtype=torch.float32
    )
    logger.info(f"Class weights: {weights_tensor.tolist()}")

    # ── Step 3: Load model ────────────────────────────────
    logger.info("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=8
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {param_count:,}")

    # ── Freeze base layers — only fine-tune last 2 transformer layers + head ─
    # Freezing layers[0-3] cuts trainable params from 67M to ~7M,
    # reducing peak RAM from ~3.5 GB to ~800 MB. Quality impact is minimal
    # because the ticket vocabulary is similar to DistilBERT pretraining data.
    for name, param in model.named_parameters():
        param.requires_grad = False  # freeze everything first

    # Unfreeze: last 2 transformer layers (layer 4 and 5 of 6)
    for name, param in model.named_parameters():
        if any(key in name for key in [
            'transformer.layer.4',
            'transformer.layer.5',
            'pre_classifier',
            'classifier',
        ]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Force garbage collection after model load
    gc.collect()

    # ── Step 4: Custom Trainer with weighted loss ─────────
    from torch.nn import CrossEntropyLoss

    class WeightedTrainer(Trainer):
        """Trainer with class-weighted cross-entropy for imbalanced datasets."""
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = CrossEntropyLoss(weight=weights_tensor.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # ── Step 5: Training ──────────────────────────────────
    # batch=1 with gradient_accumulation=16 gives effective batch=16
    # gradient_checkpointing trades compute for memory (critical on 5GB RAM)
    # Total steps = (train_samples / effective_batch) * epochs
    # 2800 / 16 * 5 = 875 steps
    total_steps = (len(tokenized_train) // 16) * 5

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'results'),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        warmup_steps=50,
        weight_decay=0.01,
        learning_rate=3e-5,
        logging_dir=os.path.join(BASE_DIR, 'logs'),
        logging_steps=25,          # Log every 25 steps (~2 min on CPU)
        evaluation_strategy="steps",
        eval_steps=50,             # Evaluate every 50 steps (~4 min)
        save_strategy="steps",
        save_steps=50,             # Must equal eval_steps when load_best_model_at_end=True
        save_total_limit=3,        # Keep 3 checkpoints (~75 steps of safety)
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=False,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=True,
        optim="adafactor",         # Much less memory than AdamW
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[MemoryProfilerCallback(total_steps=total_steps)],
    )

    logger.info("=" * 60)
    logger.info("Starting DistilBERT fine-tuning (5 epochs, weighted loss)...")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Max sequence length: {MAX_LENGTH}")
    logger.info(f"  Training samples: {len(tokenized_train)}")
    logger.info("=" * 60)
    
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        logger.info(f"Resuming training from checkpoint: {last_checkpoint}")
    else:
        logger.info("No checkpoint found. Starting from scratch.")
        
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ── Step 6: Evaluate ──────────────────────────────────
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # ── Step 7: Save ──────────────────────────────────────
    logger.info(f"Saving fine-tuned model to {MODEL_DIR}")
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Save eval results
    import json
    results_path = os.path.join(BASE_DIR, 'results', 'training_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(eval_results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    logger.info("=" * 60)
    logger.info("Training complete! Model is ready for inference.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
