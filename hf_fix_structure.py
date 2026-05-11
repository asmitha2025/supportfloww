from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()

def fix_structure():
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    repo_id = "Asmitha-28/SupportMind"
    repo_type = "space"

    root_files = [
        "model.safetensors", "config.json", "vocab.txt", 
        "tokenizer_config.json", "special_tokens_map.json", 
        "sklearn_router.pkl", "baseline_meta.json", "sla_xgb.json"
    ]
    
    print("Deleting files from root to free up quota...")
    for f in root_files:
        try:
            api.delete_file(path_in_repo=f, repo_id=repo_id, repo_type=repo_type)
            print(f"Deleted {f} from root")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

    uploads = [
        ("models/ticket_classifier/model.safetensors", "models/ticket_classifier/model.safetensors"),
        ("models/ticket_classifier/config.json", "models/ticket_classifier/config.json"),
        ("models/ticket_classifier/vocab.txt", "models/ticket_classifier/vocab.txt"),
        ("models/ticket_classifier/tokenizer_config.json", "models/ticket_classifier/tokenizer_config.json"),
        ("models/ticket_classifier/special_tokens_map.json", "models/ticket_classifier/special_tokens_map.json"),
        ("models/ticket_classifier/sklearn_router.pkl", "models/ticket_classifier/sklearn_router.pkl"),
        ("models/ticket_classifier/baseline_meta.json", "models/ticket_classifier/baseline_meta.json"),
        ("models/sla_predictor/sla_xgb.json", "models/sla_predictor/sla_xgb.json")
    ]

    print("\nUploading files to correct folders...")
    for local, repo in uploads:
        print(f"Uploading {local}...")
        try:
            api.upload_file(
                path_or_fileobj=local,
                path_in_repo=repo,
                repo_id=repo_id,
                repo_type=repo_type
            )
            print(f"Uploaded {local}")
        except Exception as e:
            print(f"Failed {local}: {e}")

    print("\nStructure fix attempt complete!")

if __name__ == "__main__":
    fix_structure()
