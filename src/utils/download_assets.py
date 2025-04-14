# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

from huggingface_hub import snapshot_download


def download_assets():
    """
    Download a KINESIS model from Hugging Face and save it in the correct format.
    
    Args:
        repo_id (str): Hugging Face repository ID (username/model-name)
        output_dir (str): Local directory to save the model
        filename (str): Name to save the model as
    
    Returns:
        str: Path to the downloaded model
    """
    data = snapshot_download(
        repo_id="amathislab/kinesis-assets",
        repo_type="dataset",
        local_dir="data",
    )

if __name__ == "__main__":
    download_assets()