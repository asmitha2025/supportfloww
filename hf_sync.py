from huggingface_hub import HfApi
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def sync_to_hf():
    # Token and Repo setup
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    repo_id = "Asmitha-28/SupportMind"
    repo_type = "space"

    logger.info(f"Starting sync to {repo_id}...")

    # 1. Upload Core Files
    core_files = ["requirements.txt", "Dockerfile", "README.md", ".env"]
    for f in core_files:
        if os.path.exists(f):
            try:
                api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id, repo_type=repo_type)
                logger.info(f"Uploaded: {f}")
            except Exception as e:
                logger.error(f"Failed to upload {f}: {e}")

    # 2. Upload Folders (src, dashboard, data)
    folders = ["src", "dashboard", "data"]
    for folder in folders:
        if os.path.exists(folder):
            logger.info(f"Uploading folder: {folder}...")
            try:
                api.upload_folder(
                    folder_path=folder,
                    path_in_repo=folder,
                    repo_id=repo_id,
                    repo_type=repo_type
                )
                logger.info(f"Folder uploaded: {folder}")
            except Exception as e:
                logger.error(f"Failed to upload folder {folder}: {e}")

    # 3. Upload Models (THE CRITICAL PART)
    # We upload each model directory separately to ensure they end up in models/
    model_dirs = [
        "models/deberta_ultimate",
        "models/ticket_classifier",
        "models/sla_predictor",
        "models/churn_signal"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            logger.info(f"Uploading model: {model_dir}...")
            try:
                api.upload_folder(
                    folder_path=model_dir,
                    path_in_repo=model_dir,
                    repo_id=repo_id,
                    repo_type=repo_type
                )
                logger.info(f"Model uploaded: {model_dir}")
            except Exception as e:
                logger.error(f"Failed to upload model {model_dir}: {e}")
        else:
            logger.warning(f"Model directory not found: {model_dir}")

    logger.info("✅ Full synchronization to Hugging Face complete!")

if __name__ == "__main__":
    sync_to_hf()
