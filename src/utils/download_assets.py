# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import argparse
from huggingface_hub import snapshot_download


def download_assets(branch="main"):
    """
    Download KINESIS assets from Hugging Face.
    
    Args:
        branch (str): Branch to download from. Defaults to "main".
    """
    data = snapshot_download(
        repo_id="amathislab/kinesis-assets",
        repo_type="dataset",
        local_dir="data",
        revision=branch,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download KINESIS assets from Hugging Face")
    parser.add_argument("--branch", type=str, default="main", help="Branch to download from")
    args = parser.parse_args()
    
    download_assets(branch=args.branch)