import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_to_onnx():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, 'models', 'deberta_ultimate')
    onnx_path = os.path.join(model_dir, 'model.onnx')

    logger.info(f"Loading PyTorch model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Put model in eval mode generally, but we will force dropout to remain via export params
    model.eval()

    # Dummy input
    text = "This is a test ticket for ONNX export."
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    logger.info("Exporting to ONNX with TrainingMode.TRAINING to preserve MC Dropout...")
    
    # We must use TrainingMode.TRAINING to keep the Dropout layers active for Monte Carlo sampling
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        export_params=True,
        opset_version=14,
        training=torch.onnx.TrainingMode.TRAINING, 
        do_constant_folding=False,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )

    logger.info(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
