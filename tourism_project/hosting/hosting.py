from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

SPACE_REPO = "dhanapalpalanisamy/tourism-app"

api = HfApi(token=os.getenv("HF_TOKEN"))

# 1. Check if Space exists
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating new Space...")
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        private=False,
        space_sdk="docker"
    )
    print(f"Space '{SPACE_REPO}' created successfully.")

# 2. Upload deployment files
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo=""
)

print("Deployment files successfully pushed to Hugging Face Space")
