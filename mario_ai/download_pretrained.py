"""
Download and adapt tsilva's pre-trained Mario checkpoints.
These were trained with skip=4; we fine-tune with skip=2 for finer control.

Usage:
    python download_pretrained.py --level 1-1   # best starting point
    python download_pretrained.py --level 1-2
"""
import argparse
import os
import urllib.request

MODELS = {
    "1-1": "https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-1_gray84-hudcrop-stack4-simple_ppo/resolve/main/model.zip",
    "1-2": "https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-2_gray84-hudcrop-stack4-simple_ppo/resolve/main/model.zip",
    "1-4": "https://huggingface.co/tsilva/NES-SuperMarioBros_Level1-4_gray84-hudcrop-stack4-simple_ppo/resolve/main/model.zip",
    "2-1": "https://huggingface.co/tsilva/NES-SuperMarioBros_Level2-1_gray84-hudcrop-stack4-simple_ppo/resolve/main/model.zip",
    "3-2": "https://huggingface.co/tsilva/NES-SuperMarioBros_Level3-2_gray84-hudcrop-stack4-simple_ppo/resolve/main/model.zip",
}


def download(level: str, out_dir: str = "pretrained"):
    if level not in MODELS:
        print(f"Unknown level {level}. Available: {list(MODELS.keys())}")
        return None

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"mario_{level.replace('-', '_')}_pretrained.zip")

    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return dest

    url = MODELS[level]
    print(f"Downloading level {level} checkpoint (~21MB)...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\nSaved to {dest}")
    return dest


def _progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\r  {pct}%", end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="1-1", choices=list(MODELS.keys()))
    args = parser.parse_args()
    download(args.level)
