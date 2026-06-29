import os
import requests
import sys

# Configuration: List of models to download
# We use Hugging Face mirrors for both as they are more stable/accessible than GitHub
MODELS = [
    {
        "filename": "groundingdino_swint_ogc.pth",
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
        "min_size_mb": 150,  # Real size is ~172 MB
    },
    {
        "filename": "groundingdino_swinb_cogcoor.pth",
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
        "min_size_mb": 800,  # Real size is ~894 MB
    },
]


def download_file(url, dest_path, min_size_mb):
    print(f"\n--- Processing {os.path.basename(dest_path)} ---")

    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        if size_mb > min_size_mb:
            print(f"Model already exists ({size_mb:.2f} MB). Skipping.")
            return
        else:
            print(f"Found corrupt/partial file ({size_mb:.6f} MB). Re-downloading...")
            os.remove(dest_path)

    try:
        print(f"Source: {url}")
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            downloaded = 0
            for i, chunk in enumerate(response.iter_content(chunk_size=8192)):
                f.write(chunk)
                downloaded += len(chunk)
                if i % 2000 == 0:
                    print(
                        f"   Downloading... {downloaded / (1024 * 1024):.0f} MB",
                        end="\r",
                    )

        final_size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        if final_size_mb < min_size_mb:
            print(f"\nError: File too small ({final_size_mb:.2f} MB). Download failed.")
            os.remove(dest_path)
            sys.exit(1)

        print(f"\nSuccess! Saved ({final_size_mb:.2f} MB)")

    except Exception as e:
        print(f"\nDownload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Checking {len(MODELS)} GroundingDINO models...")

    for model_config in MODELS:
        target_path = os.path.join(script_dir, model_config["filename"])
        download_file(model_config["url"], target_path, model_config["min_size_mb"])
