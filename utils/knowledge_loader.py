"""
utils/knowledge_loader.py

Downloads the private knowledge base from a HuggingFace Dataset at startup.
This keeps personal knowledge files OUT of the public repo (GitHub + HF Space)
while still making them available inside the running app.

Setup (HuggingFace Space Secrets):
  HF_TOKEN              → your HF read token (huggingface.co/settings/tokens)
  HF_KNOWLEDGE_DATASET  → e.g. "kuldeepmishra3/kuldeep-knowledge"

Locally: just keep files in data/knowledge/ as normal (no download needed).
"""

import os
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

KNOWLEDGE_DIR = Path("data/knowledge")


def download_knowledge_from_hf() -> bool:
    """
    Download knowledge files from a private HF Dataset.
    Only runs when HF_TOKEN and HF_KNOWLEDGE_DATASET are set (i.e. on HF Space).
    Returns True if downloaded, False if skipped (local mode).
    """
    hf_token   = os.environ.get("HF_TOKEN", "").strip()
    hf_dataset = os.environ.get("HF_KNOWLEDGE_DATASET", "").strip()

    if not hf_token or not hf_dataset:
        logger.info("[KnowledgeLoader] HF_TOKEN or HF_KNOWLEDGE_DATASET not set — using local files.")
        return False

    try:
        from huggingface_hub import snapshot_download

        logger.info(f"[KnowledgeLoader] Downloading knowledge base from private dataset: {hf_dataset}")
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=hf_dataset,
            repo_type="dataset",
            local_dir=str(KNOWLEDGE_DIR),
            token=hf_token,
            ignore_patterns=["*.gitattributes", ".gitattributes", "README.md"],
        )

        files = list(KNOWLEDGE_DIR.glob("*.txt")) + list(KNOWLEDGE_DIR.glob("*.pdf"))
        logger.info(f"[KnowledgeLoader] ✓ Downloaded {len(files)} knowledge file(s): {[f.name for f in files]}")
        return True

    except Exception as e:
        logger.error(f"[KnowledgeLoader] ✗ Failed to download knowledge base: {e}")
        return False
