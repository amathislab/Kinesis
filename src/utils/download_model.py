# Copyright (c) 2025 Mathis Group for Computational Neuroscience and AI, EPFL
# All rights reserved.
#
# Licensed under the BSD 3-Clause License.

import argparse
import os
from huggingface_hub import hf_hub_download


def download_kinesis_model(
    repo_id,
    output_dir,
):
    """
    Download a KINESIS model from Hugging Face and save it in the correct format.
    
    Args:
        repo_id (str): Hugging Face repository ID (username/model-name)
        output_dir (str): Local directory to save the model
        filename (str): Name to save the model as
    
    Returns:
        str: Path to the downloaded model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the model
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pth", local_dir=output_dir)
    
    print(f"Model downloaded and saved to {output_dir}")
    
    return model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a KINESIS model from Hugging Face")
    parser.add_argument("--repo_id", type=str, help="Hugging Face repository ID (username/model-name)")

    args = parser.parse_args()

    output_dir = "data/trained_models/" + args.repo_id.split("/")[-1]
    # Example usage
    download_kinesis_model(
        repo_id=args.repo_id,
        output_dir=output_dir
    )