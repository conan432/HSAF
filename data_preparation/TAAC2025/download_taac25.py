from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO_ID = "salmon1802/TAAC-25"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "UniRank-TAAC-25"
DEFAULT_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--local-dir",
        default=str(DEFAULT_LOCAL_DIR),
        help="Where to place the downloaded dataset repository.",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help="Hugging Face endpoint mirror. Default: hf-mirror.com.",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision/branch/tag.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel download workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with: pip install -U 'huggingface_hub[cli]'"
        ) from exc

    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_id    : {args.repo_id}")
    print(f"endpoint   : {args.endpoint}")
    print(f"local_dir  : {local_dir}")
    print(f"revision   : {args.revision}")
    print(f"workers    : {args.workers}")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
        max_workers=args.workers,
        endpoint=args.endpoint,
    )
    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()

