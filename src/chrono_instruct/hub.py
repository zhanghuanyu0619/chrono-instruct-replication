"""Push a trained vintage to the Hugging Face Hub.

Used by `chrono push` and, when `push_to_hub.enabled` is set, at the end of
training. Needs a write token: run `hf auth login` or set HF_TOKEN. Never
hardcode the token (it would leak via git and get auto-revoked).
"""
import os

from huggingface_hub import HfApi


def push_dir(local_dir, repo_id, private=True, token=None, commit_message=None):
    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model",
                      commit_message=commit_message or "upload checkpoint")
    print(f"Pushed {local_dir} -> https://huggingface.co/{repo_id}")
