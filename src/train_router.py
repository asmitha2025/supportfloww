# src/train_router.py
# Fine-tune DistilBERT for 8-class ticket routing
# SupportMind v1.0 — Asmitha
#
# Memory-optimized for machines with limited RAM:
#   - max_length=128 (vs 256) — tickets are short
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
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import pandas as pd
import torch
import logging
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'ticket_classifier')

# Shorter max_length — support tickets are typically short
MAX_LENGTH = 128


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

    # ── Step 2: Load model ────────────────────────────────
    logger.info("Loading DistilBERT model (low_cpu_mem_usage=True)...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=8
    )
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {param_count:,}")

    # ── Step 3: Training ──────────────────────────────────
    # Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
    # = 2 × 8 = 16 (good effective batch without memory spike)
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, 'results'),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=os.path.join(BASE_DIR, 'logs'),
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=use_fp16,
        dataloader_num_workers=0,  # Avoid multiprocessing memory overhead on Windows
        report_to="none",  # Disable wandb/tensorboard to save memory
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    logger.info("=" * 60)
    logger.info("Starting DistilBERT fine-tuning (3 epochs)...")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Max sequence length: {MAX_LENGTH}")
    logger.info(f"  Training samples: {len(tokenized_train)}")
    logger.info("=" * 60)
    
    trainer.train()

    # ── Step 4: Evaluate ──────────────────────────────────
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Eval results: {eval_results}")

    # ── Step 5: Save ──────────────────────────────────────
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


if __name__ == "__main__":
    main()

