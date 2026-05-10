import os
import sys
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

def train_baseline():
    """
    Trains a TF-IDF + CalibratedClassifierCV(LogisticRegression) pipeline
    and saves it as models/ticket_classifier/sklearn_router.pkl.
    """
    print("Training TF-IDF + Logistic Regression baseline pipeline...")
    
    # Ensure working directory is project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    data_path = os.path.join("data", "raw", "support_tickets.csv")
    
    # Generate some fallback synthetic data if CSV is not present
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        try:
            df = pd.read_csv(data_path)
            # Assuming columns 'text' and 'category' exist
            if 'text' in df.columns and 'category' in df.columns:
                texts = df['text'].dropna().astype(str).tolist()
                labels = df['category'].dropna().astype(str).tolist()
            else:
                raise ValueError("CSV missing 'text' or 'category' columns.")
        except Exception as e:
            print(f"Error reading CSV: {e}. Falling back to synthetic data.")
            texts, labels = get_synthetic_data()
    else:
        print(f"{data_path} not found. Generating synthetic baseline data...")
        texts, labels = get_synthetic_data()

    print(f"Training on {len(texts)} samples...")
    
    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', CalibratedClassifierCV(LogisticRegression(class_weight='balanced', max_iter=1000))),
    ])

    # Fit the pipeline
    pipeline.fit(texts, labels)

    # Save the model
    out_dir = os.path.join("models", "ticket_classifier")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sklearn_router.pkl")
    
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Baseline model successfully saved to {out_path}")

def get_synthetic_data():
    """Returns synthetic data for fallback training."""
    categories = ['billing', 'technical_support', 'sales', 'account_management']
    base_texts = {
        'billing': ["invoice is wrong", "charge on my card", "cancel subscription", "refund request", "pricing plan"],
        'technical_support': ["server is down", "cannot login", "getting 500 error", "bug in the app", "export failing"],
        'sales': ["want to upgrade", "talk to sales", "enterprise pricing", "demo request", "more seats"],
        'account_management': ["change password", "update email", "delete account", "add user", "role permissions"]
    }
    
    texts = []
    labels = []
    
    # Create a reasonably sized synthetic dataset
    for cat in categories:
        for _ in range(50):
            for text in base_texts[cat]:
                texts.append(text)
                labels.append(cat)
                
    return texts, labels

if __name__ == "__main__":
    train_baseline()
