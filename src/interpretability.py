# src/interpretability.py
# Model Interpretability Module — SHAP Explanations
# SupportMind v1.0 — Asmitha

import torch
import shap
import numpy as np
from transformers import pipeline
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class SupportMindExplainer:
    """
    Provides SHAP-based explanations for DistilBERT predictions.
    
    Helps support agents understand WHY a ticket was routed to a specific
    category by highlighting the most influential words.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Create a transformers pipeline for SHAP
        self.pipe = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == 'cuda' else -1,
            top_k=None, # Get all class probabilities
            framework="pt" # Force PyTorch
        )

        
        # Initialize SHAP explainer
        # We use a simple wrap to make it compatible with SHAP's expectations
        def predictor(texts):
            # Convert numpy array to list if necessary for transformers pipeline
            if isinstance(texts, np.ndarray):
                texts = texts.tolist()
            outputs = self.pipe(texts, batch_size=32)
            # SHAP expects a matrix of [num_samples, num_classes]
            # Outputs is a list of lists of dicts: [[{'label': 'LABEL_0', 'score': 0.1}, ...], ...]
            # We need to ensure the order matches the CATEGORY_MAP
            from confidence_router import CATEGORY_MAP, CATEGORY_REVERSE
            
            num_classes = len(CATEGORY_MAP)
            results = np.zeros((len(texts), num_classes))
            
            for i, out in enumerate(outputs):
                for item in out:
                    # HF labels are usually 'LABEL_N' or the actual category names
                    label = item['label']
                    score = item['score']
                    
                    # If label is 'LABEL_N'
                    if label.startswith('LABEL_'):
                        idx = int(label.split('_')[1])
                        results[i, idx] = score
                    # If label is the category name
                    elif label in CATEGORY_REVERSE:
                        idx = CATEGORY_REVERSE[label]
                        results[i, idx] = score
                        
            return results

        # SHAP Explainer for text
        # Using a small masker for performance
        self.explainer = shap.Explainer(predictor, self.tokenizer)

    def explain(self, text: str, target_class_idx: int = None) -> Dict[str, Any]:
        """
        Generate SHAP values for a single ticket.
        
        Args:
            text: The ticket text.
            target_class_idx: The class index to explain. If None, uses the predicted class.
            
        Returns:
            Dictionary with tokens and their corresponding SHAP values.
        """
        try:
            # Generate SHAP values
            # This can be slow for long texts, but for tickets (~128 tokens) it's manageable
            shap_values = self.explainer([text])
            
            # If target_class_idx is not provided, use the one with highest mean SHAP value
            if target_class_idx is None:
                # shap_values.values has shape [samples, tokens, classes]
                # We take the class that has the highest average value for this sample
                target_class_idx = np.argmax(np.mean(np.abs(shap_values.values[0]), axis=0))

            # Extract tokens and values for the target class
            # shap_values[sample_idx, :, class_idx]
            values = shap_values.values[0, :, target_class_idx]
            base_value = float(shap_values.base_values[0, target_class_idx])
            
            # SHAP returns tokens as they are produced by the tokenizer (e.g. '##ing', ' [CLS]')
            # We want to map these back to something readable if possible, but raw tokens are okay for highlighting
            tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            
            # Filter out special tokens like [CLS], [SEP], [PAD] for the final output
            # but keep the alignment
            result_tokens = []
            result_values = []
            
            for token, val in zip(tokens, values):
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue
                result_tokens.append(token)
                result_values.append(float(val))

            return {
                'tokens': result_tokens,
                'values': result_values,
                'base_value': base_value,
                'target_class': target_class_idx,
                'prediction_value': float(base_value + np.sum(values))
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {'error': str(e)}

if __name__ == '__main__':
    # Test
    from confidence_router import ConfidenceGatedRouter
    router = ConfidenceGatedRouter()
    explainer = SupportMindExplainer(router.model, router.tokenizer)
    
    test_text = "My invoice is wrong, please fix the billing error."
    res = explainer.explain(test_text)
    print(f"Tokens: {res['tokens']}")
    print(f"Values: {res['values']}")
