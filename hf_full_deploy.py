from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()

SKIP_DIRS = {'__pycache__', '.pytest_cache', 'logs', 'scratch'}
SKIP_SUFFIXES = ('.pyc', '.pyo', '.log')
SKIP_FILES = {'src/server_log.txt'}


def should_upload(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = set(normalized.split("/"))
    if parts & SKIP_DIRS:
        return False
    if normalized in SKIP_FILES:
        return False
    return not normalized.endswith(SKIP_SUFFIXES)


def full_deploy():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Add it as a local environment variable or Space secret.")
    api = HfApi(token=token)
    repo_id = "Asmitha-28/SupportMind"
    repo_type = "space"

    print("Uploading root files...")
    root_files = ["requirements.txt", "Dockerfile", "README.md", ".env.example"]
    for f in root_files:
        if os.path.exists(f):
            try:
                api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo_id, repo_type=repo_type)
                print(f"Uploaded {f}")
            except Exception as e:
                print(f"Failed {f}: {e}")

    print("\nUploading src/ directory...")
    if os.path.exists("src"):
        for f in os.listdir("src"):
            path = os.path.join("src", f)
            if os.path.isfile(path) and should_upload(path):
                repo_path = path.replace("\\", "/")
                try:
                    api.upload_file(path_or_fileobj=path, path_in_repo=repo_path, repo_id=repo_id, repo_type=repo_type)
                    print(f"Uploaded {repo_path}")
                except Exception as e:
                    print(f"Failed {path}: {e}")

    print("\nUploading dashboard/ directory...")
    if os.path.exists("dashboard"):
        for root, dirs, files in os.walk("dashboard"):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                path = os.path.join(root, f)
                if not should_upload(path):
                    continue
                try:
                    api.upload_file(path_or_fileobj=path, path_in_repo=path.replace("\\", "/"), repo_id=repo_id, repo_type=repo_type)
                    print(f"Uploaded {path}")
                except Exception as e:
                    print(f"Failed {path}: {e}")

    print("\nUploading data files...")
    if os.path.exists("data/clarification_bank.json"):
        try:
            api.upload_file(path_or_fileobj="data/clarification_bank.json", path_in_repo="data/clarification_bank.json", repo_id=repo_id, repo_type=repo_type)
            print("Uploaded data/clarification_bank.json")
        except Exception as e:
            print(f"Failed data/clarification_bank.json: {e}")

    print("\nDeployment complete!")

if __name__ == "__main__":
    full_deploy()
