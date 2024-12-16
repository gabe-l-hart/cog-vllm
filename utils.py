import os
import requests
import subprocess
import time
import warnings
from urllib.parse import urlparse
from huggingface_hub import model_info, snapshot_download


HF_TOKEN = os.getenv("HF_TOKEN")


async def resolve_model_path(url_or_local_path: str) -> str:
    """
    Resolves the model path, downloading if necessary.

    Args:
        url_or_local_path (str): URL to the tarball or local path to a directory containing the model artifacts.

    Returns:
        str: Path to the directory containing the model artifacts.
    """

    parsed_url = urlparse(url_or_local_path)
    if parsed_url.scheme == "http" or parsed_url.scheme == "https":
        return await download_tarball(url_or_local_path)
    elif parsed_url.scheme == "file" or parsed_url.scheme == "":
        if not os.path.exists(parsed_url.path):
            raise ValueError(
                f"E1000: The provided local path '{parsed_url.path}' does not exist."
            )
        if not os.listdir(parsed_url.path):
            raise ValueError(
                f"E1000: The provided local path '{parsed_url.path}' is empty."
            )

        warnings.warn(
            "Using local model artifacts for development is okay, but not optimal for production. "
            "To minimize boot time, store model assets externally on Replicate."
        )
        return url_or_local_path
    elif is_hf_model(url_or_local_path):
        return snapshot_download(url_or_local_path)
    else:
        raise ValueError(f"E1000: Unsupported model path scheme: {parsed_url.scheme}")


def is_hf_model(model_name: str) -> bool:
    """Check whether the given model name is a valid HF model"""
    try:
        model_info(model_name, token=HF_TOKEN)
        return True
    except requests.exceptions.HTTPError:
        return False


def download_huggingface(model_name: str) -> str:
    """
    Downloads a Hugging Face model from the Hub

    Args:
        model_name (str): Name of the Hugging Face model to download.

    Returns:
        str: Path to the directory where the model was extracted.
    """
    filename = model_name.split("/")[-1]
    models_dir = os.path.join(os.getcwd(), "models")
    path = os.path.join(models_dir, filename)
    if os.path.exists(path) and os.listdir(path):
        print(f"Files already present in `{path}`.")
        return path
    os.makedirs(models_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        token=HF_TOKEN,
        local_dir=filename,
    )
    return filename


async def download_tarball(url: str) -> str:
    """
    Downloads a tarball from a URL and extracts it.

    Args:
        url (str): URL to the tarball.

    Returns:
        str: Path to the directory where the tarball was extracted.
    """
    filename = os.path.splitext(os.path.basename(url))[0]
    path = os.path.join(os.getcwd(), "models", filename)
    if os.path.exists(path) and os.listdir(path):
        print(f"Files already present in `{path}`.")
        return path

    print(f"Downloading model assets to {path}...")
    start_time = time.time()
    command = ["pget", url, path, "-x"]
    subprocess.check_call(command, close_fds=True)
    print(f"Downloaded model assets in {time.time() - start_time:.2f}s")
    return path
