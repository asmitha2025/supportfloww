from huggingface_hub import HfApi
import os
import sys
from dotenv import load_dotenv
load_dotenv()

def deploy():
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set. Add it as a local environment variable or Space secret.")
        sys.exit(1)
    api = HfApi(token=token)

    try:
        username = api.whoami()["name"]
    except Exception as e:
        print(f"Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)
        
    repo_id = f"{username}/SupportMind"

    print(f"Creating Space {repo_id}...")
    try:
        api.delete_repo(repo_id=repo_id, repo_type="space")
        print("Cleared old space allocation.")
    except Exception:
        pass

    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    print("Uploading minimal project files to Hugging Face Spaces...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            allow_patterns=[
                "src/**",
                "dashboard/**",
                "data/clarification_bank.json",
                "requirements.txt",
                "Dockerfile",
                "README.md",
                ".env.example"
            ]
        )
        print(f"✅ Code successfully deployed! Now the models are in the root on HF, we need to move them to the correct folders.")
        print(f"Your live demo will be available at: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Failed to upload files: {e}")

if __name__ == "__main__":
    deploy()
