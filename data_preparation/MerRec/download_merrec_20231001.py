#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Download the 20231001 split of mercari-us/merrec from Hugging Face.

Default output:
  data/MerRec/20231001/

Usage:
  python data/MerRec/download_merrec_20231001.py
  python data/MerRec/download_merrec_20231001.py --local-dir /path/to/MerRec
  HF_TOKEN=xxx python data/MerRec/download_merrec_20231001.py

If huggingface.co is slow or blocked, try a mirror:
  HF_ENDPOINT=https://hf-mirror.com python data/MerRec/download_merrec_20231001.py
"""

import argparse
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urljoin

from tqdm.auto import tqdm

try:
    from huggingface_hub import HfApi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with: "
        "pip install huggingface_hub tqdm"
    ) from exc


DEFAULT_REPO_ID = "mercari-us/merrec"
DEFAULT_FOLDER = "20231001"
DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a specific folder from the mercari-us/merrec Hugging Face dataset."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face dataset repo id.")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="Folder to download from the dataset repo.")
    parser.add_argument(
        "--local-dir",
        default=str(Path(__file__).resolve().parent),
        help="Local MerRec directory. The selected folder will be saved under this directory.",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision/branch/tag.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_ENDPOINT),
        help="Hugging Face endpoint, e.g. https://huggingface.co or https://hf-mirror.com.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN environment variable.",
    )
    parser.add_argument("--force-download", action="store_true", help="Re-download files even if they exist locally.")
    parser.add_argument("--dry-run", action="store_true", help="Only list matched files without downloading.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for each file.")
    return parser.parse_args()


def list_folder_files(repo_id, folder, revision, token, endpoint):
    api = HfApi(endpoint=endpoint, token=token)
    repo_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
    )
    prefix = folder.strip("/") + "/"
    return sorted(path for path in repo_files if path.startswith(prefix) and not path.endswith("/"))


def build_resolve_url(endpoint, repo_id, revision, filename):
    endpoint = endpoint.rstrip("/")
    repo_id = quote(repo_id.strip("/"), safe="/")
    revision = quote(revision, safe="")
    filename = quote(filename, safe="/")
    return f"{endpoint}/datasets/{repo_id}/resolve/{revision}/{filename}"


def make_request(url, token):
    headers = {"User-Agent": "UniRank-MerRec-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def open_with_redirects(url, token, timeout=60, max_redirects=10):
    current_url = url
    for _ in range(max_redirects + 1):
        request = make_request(current_url, token)
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in {301, 302, 303, 307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            next_url = urljoin(current_url, location)
            if next_url == current_url:
                raise RuntimeError(f"Redirect loop detected for {current_url}") from exc
            current_url = next_url
    raise RuntimeError(f"Too many redirects for {url}")


def download_file(url, output_path, token, force_download=False, retries=3):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0 and not force_download:
        tqdm.write(f"Skip existing: {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".incomplete")
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            with open_with_redirects(url, token, timeout=60) as response:
                total = response.headers.get("Content-Length")
                total = int(total) if total and total.isdigit() else None
                desc = output_path.name
                with tmp_path.open("wb") as fout, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    leave=False,
                ) as progress:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        fout.write(chunk)
                        progress.update(len(chunk))
            tmp_path.replace(output_path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                tqdm.write(f"Retry {attempt}/{retries} for {output_path.name}: {exc}")
                time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Failed to download {url}: {last_error}")


def main():
    args = parse_args()
    local_dir = Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    files = list_folder_files(args.repo_id, args.folder, args.revision, args.token, args.endpoint)
    if not files:
        raise SystemExit(
            f"No files found under hf://datasets/{args.repo_id}/{args.folder}/ "
            f"at revision {args.revision}."
        )

    target_dir = local_dir / args.folder.strip("/")
    print(f"Repository : hf://datasets/{args.repo_id}")
    print(f"Endpoint   : {args.endpoint}")
    print(f"Folder     : {args.folder}")
    print(f"Files      : {len(files)}")
    print(f"Output     : {target_dir}")

    if args.dry_run:
        for path in files:
            print(path)
        return

    for filename in tqdm(files, desc="Downloading MerRec 20231001", unit="file"):
        url = build_resolve_url(args.endpoint, args.repo_id, args.revision, filename)
        output_path = local_dir / filename
        download_file(
            url=url,
            output_path=output_path,
            token=args.token,
            force_download=args.force_download,
            retries=args.retries,
        )

    print(f"Done. Downloaded files are available in: {target_dir}")


if __name__ == "__main__":
    main()
