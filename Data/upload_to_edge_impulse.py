"""
upload_to_edge_impulse.py

Upload the SENTINEL-X CNN crop dataset to Edge Impulse.

The script reads Data/CNN-Data/labels.csv and sends:
    train + validation -> Edge Impulse training
    test               -> Edge Impulse testing

The Edge Impulse project API key is read from .env as EI_Key. The key is never
printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib import error, request
from urllib.parse import urlencode


DATA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DATA_DIR.parent
CNN_DATA_DIR = DATA_DIR / "CNN-Data"
LABELS_CSV = CNN_DATA_DIR / "labels.csv"
ENV_FILE = PROJECT_DIR / ".env"
EDGE_IMPULSE_CONFIG = Path.home() / "edge-impulse-config.json"
EDGE_IMPULSE_API = "https://studio.edgeimpulse.com/v1"

DEFAULT_KEY_NAME = "EI_Key"
DEFAULT_CHUNK_SIZE = 50
DEFAULT_PROJECT_NAME = "Sentinel-X"
DEFAULT_HMAC_KEY_NAMES = ("EI_HMAC_Key", "HMAC_Key", "EDGE_IMPULSE_HMAC_KEY", "EI_HMAC")


def uploader_command() -> list[str]:
    """Return a Windows-friendly Edge Impulse uploader command."""
    for command_name in ("edge-impulse-uploader.cmd", "edge-impulse-uploader.exe", "edge-impulse-uploader"):
        command_path = shutil.which(command_name)
        if command_path:
            return [command_path]

    ps1_path = shutil.which("edge-impulse-uploader.ps1")
    if ps1_path:
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps1_path]

    raise FileNotFoundError("Could not find edge-impulse-uploader on PATH")


def split_env_line(line: str) -> tuple[str, str] | None:
    """Split common .env assignment styles into key and value."""
    clean = line.strip()
    if clean.startswith("export "):
        clean = clean.removeprefix("export ").strip()

    if "=" in clean:
        return clean.split("=", 1)
    if ":" in clean:
        return clean.split(":", 1)

    parts = re.split(r"\s+", clean, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]

    return None


def normalize_env_name(name: str) -> str:
    """Normalize env names so EI_Key, EI-Key, and EI Key can match."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def read_env_value(name: str) -> str | None:
    """Read one value from .env without printing the file contents."""
    value = os.environ.get(name)
    if value:
        return value

    if not ENV_FILE.is_file():
        return None

    wanted = normalize_env_name(name)
    unnamed_lines: list[str] = []
    with ENV_FILE.open(encoding="utf-8-sig") as f:
        for line in f:
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            pair = split_env_line(clean)
            if pair is None:
                unnamed_lines.append(clean)
                continue

            key, raw_value = pair
            if normalize_env_name(key.strip()) != wanted:
                continue

            return raw_value.strip().strip('"').strip("'")

    # Some local setups store only the raw project API key on one line.
    if len(unnamed_lines) == 1:
        return unnamed_lines[0].strip('"').strip("'")

    return None


def edge_api_get(api_key: str, api_path: str) -> dict:
    """Call the Edge Impulse Studio API without logging the key."""
    return edge_api_request(api_key, "GET", api_path)


def edge_api_post(api_key: str, api_path: str, payload: dict) -> dict:
    """POST to the Edge Impulse Studio API without logging secrets."""
    return edge_api_request(api_key, "POST", api_path, payload)


def edge_api_request(api_key: str, method: str, api_path: str, payload: dict | None = None) -> dict:
    """Call the Edge Impulse Studio API without logging the key."""
    url = f"{EDGE_IMPULSE_API}{api_path}"
    body = None
    headers = {
        "accept": "application/json",
        "x-api-key": api_key,
    }
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"

    api_request = request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with request.urlopen(api_request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Edge Impulse API request failed ({exc.code}): {body}") from exc


def find_project(api_key: str, project_name: str) -> dict:
    """Find the named Edge Impulse project for an account-level key."""
    data = edge_api_get(api_key, "/api/projects")
    projects = data.get("projects", [])
    matches = [project for project in projects if project.get("name", "").casefold() == project_name.casefold()]

    if not matches:
        names = ", ".join(sorted(project.get("name", "") for project in projects if project.get("name")))
        raise RuntimeError(f"Could not find Edge Impulse project '{project_name}'. Visible projects: {names}")
    if len(matches) > 1:
        raise RuntimeError(f"Found multiple Edge Impulse projects named '{project_name}'. Rename one or use one account.")

    return matches[0]


def read_edge_impulse_config() -> dict:
    """Load the Edge Impulse CLI config while preserving unrelated settings."""
    if not EDGE_IMPULSE_CONFIG.is_file():
        return {}

    with EDGE_IMPULSE_CONFIG.open(encoding="utf-8") as f:
        return json.load(f)


def set_uploader_project(project_id: int) -> None:
    """Cache the uploader project ID so the CLI does not prompt."""
    config = read_edge_impulse_config()
    config["uploaderProjectId"] = project_id

    temp_path = EDGE_IMPULSE_CONFIG.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    temp_path.replace(EDGE_IMPULSE_CONFIG)


def resolve_project_api_key(account_key: str, project_name: str, project_id: int | None) -> tuple[str, int]:
    """Resolve a project development key for the named Edge Impulse project."""
    if project_id is not None:
        set_uploader_project(project_id)
        print(f"Using Edge Impulse project ID: {project_id}")
        return account_key, project_id

    project = find_project(account_key, project_name)
    project_id = int(project["id"])
    set_uploader_project(project_id)

    devkeys = edge_api_get(account_key, f"/api/{project_id}/devkeys")
    project_key = devkeys.get("apiKey")
    if not project_key:
        print(f"Project '{project_name}' has no development API key. Using the provided API key for upload.")
        return account_key, project_id

    print(f"Resolved Edge Impulse project: {project_name} (ID {project_id})")
    return project_key, project_id


def local_filename_map() -> dict[str, tuple[str, str]]:
    """Map local crop filename stems to Edge Impulse category and label."""
    mapping: dict[str, tuple[str, str]] = {}
    with LABELS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["crop_file"]).stem
            mapping[stem] = (edge_category(row["split"]), row["label"])
    return mapping


def list_edge_samples(api_key: str, project_id: int, category: str) -> list[dict]:
    """List all Edge Impulse samples in one category."""
    samples: list[dict] = []
    offset = 0
    limit = 1000

    while True:
        query = urlencode({"category": category, "limit": limit, "offset": offset})
        data = edge_api_get(api_key, f"/api/{project_id}/raw-data?{query}")
        batch = data.get("samples", [])
        samples.extend(batch)
        if len(batch) < limit:
            return samples
        offset += limit


def repair_uploaded_labels(api_key: str, project_id: int) -> None:
    """Switch the project to single-label mode and repair uploaded sample labels."""
    print("Setting project labeling method to single_label ...")
    edge_api_post(api_key, f"/api/{project_id}", {"labelingMethod": "single_label"})

    filename_map = local_filename_map()
    repair_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    missing = 0

    for category in ("testing", "training"):
        samples = list_edge_samples(api_key, project_id, category)
        print(f"Checking {len(samples)} uploaded {category} samples ...")
        for sample in samples:
            expected = filename_map.get(sample["filename"])
            if expected is None:
                missing += 1
                continue

            expected_category, expected_label = expected
            if expected_category != category:
                continue
            if sample.get("label") == expected_label:
                continue

            repair_groups[(category, expected_label)].append(int(sample["id"]))

    for (category, label), ids in sorted(repair_groups.items()):
        for batch in chunks(ids, 500):
            query = urlencode({"category": category, "ids": json.dumps(batch)})
            print(f"Repairing {len(batch)} labels: {category} / {label}")
            edge_api_post(api_key, f"/api/{project_id}/raw-data/batch/edit-labels?{query}", {"label": label})

    print(f"Label repair complete. Unmapped uploaded samples: {missing}")


def edge_category(split: str) -> str:
    """Map local split names to Edge Impulse categories."""
    if split in {"train", "validation"}:
        return "training"
    if split == "test":
        return "testing"
    raise ValueError(f"Unsupported split: {split}")


def read_samples(limit: int | None) -> dict[tuple[str, str], list[Path]]:
    """Group crop paths by Edge Impulse category and label."""
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"Missing {LABELS_CSV}")

    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)
    with LABELS_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = edge_category(row["split"])
            label = row["label"]
            crop_path = (CNN_DATA_DIR / row["crop_file"]).resolve()

            # Refuse paths that escape CNN-Data.
            crop_path.relative_to(CNN_DATA_DIR.resolve())
            if crop_path.is_file():
                grouped[(category, label)].append(crop_path)

            if limit is not None and total_count(grouped) >= limit:
                break

    return dict(grouped)


def total_count(grouped: dict[tuple[str, str], list[Path]]) -> int:
    """Count every sample in a grouped mapping."""
    return sum(len(paths) for paths in grouped.values())


def chunks(paths: list[Path], size: int) -> list[list[Path]]:
    """Split a path list into command-sized chunks."""
    return [paths[i : i + size] for i in range(0, len(paths), size)]


def print_summary(grouped: dict[tuple[str, str], list[Path]]) -> None:
    """Print counts without showing secrets."""
    print(f"Samples ready: {total_count(grouped)}")
    for category, label in sorted(grouped):
        print(f"  {category:8s} {label:40s} {len(grouped[(category, label)]):7d}")


def upload_group(
    api_key: str,
    hmac_key: str | None,
    category: str,
    label: str,
    paths: list[Path],
    chunk_size: int,
    concurrency: int,
) -> None:
    """Upload one category/label group in small batches."""
    uploader = uploader_command()
    for batch_number, batch in enumerate(chunks(paths, chunk_size), start=1):
        print(f"Uploading {category} / {label}: batch {batch_number} ({len(batch)} files)")
        info = {
            "version": 1,
            "files": [
                {
                    "path": str(path),
                    "category": category,
                    # The uploader expects a label object. A plain string is ignored.
                    "label": {"type": "label", "label": label},
                }
                for path in batch
            ],
        }

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump(info, f)
            info_path = Path(f.name)

        try:
            command = [
                *uploader,
                "--api-key",
                api_key,
                "--concurrency",
                str(concurrency),
                "--info-file",
                str(info_path),
            ]
            if hmac_key:
                command.extend(["--hmac-key", hmac_key])

            result = subprocess.run(command, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout, end="")
            if result.stderr:
                print(result.stderr, end="")
            if result.returncode != 0:
                raise RuntimeError(
                    f"Edge Impulse upload failed for {category} / {label} batch {batch_number}. "
                    "The command arguments were hidden to avoid leaking secrets."
                )
        finally:
            info_path.unlink(missing_ok=True)


def plan_remote_sample(
    filename: str,
    label: str | None,
    category: str,
    desired: dict[tuple[str, str], str],
    seen: set[tuple[str, str]],
) -> tuple[str, str | None]:
    """Decide whether one uploaded sample should be kept, relabeled, or deleted."""
    key = (category, Path(filename).stem)
    wanted = desired.get(key)
    # A second copy of the same file is extra. Delete it so caps stay exact.
    if wanted is None or key in seen:
        return "delete", None
    seen.add(key)
    if label != wanted:
        return "relabel", wanted
    return "keep", wanted


def wait_for_job(api_key: str, project_id: int, payload: dict) -> None:
    """Wait when a batch call starts a background job."""
    job_id = payload.get("id")
    if not job_id:
        return
    while True:
        status = edge_api_get(api_key, f"/api/{project_id}/jobs/{job_id}/status")
        job = status.get("job", {})
        if job.get("finished"):
            if not job.get("finishedSuccessful"):
                raise RuntimeError(f"Edge Impulse job {job_id} failed.")
            return
        time.sleep(2)


def sync_dataset(
    api_key: str,
    hmac_key: str | None,
    project_id: int,
    grouped: dict[tuple[str, str], list[Path]],
    chunk_size: int,
    concurrency: int,
) -> None:
    """Make Edge Impulse match CNN-Data without re-uploading files that are already there."""
    print("Setting project labeling method to single_label ...")
    edge_api_post(api_key, f"/api/{project_id}", {"labelingMethod": "single_label"})

    desired: dict[tuple[str, str], str] = {}
    path_by_key: dict[tuple[str, str], Path] = {}
    for (category, label), paths in grouped.items():
        for path in paths:
            key = (category, path.stem)
            desired[key] = label
            path_by_key[key] = path

    delete_ids: dict[str, list[int]] = defaultdict(list)
    relabel_ids: dict[tuple[str, str], list[int]] = defaultdict(list)
    present: set[tuple[str, str]] = set()

    for category in ("training", "testing"):
        samples = list_edge_samples(api_key, project_id, category)
        print(f"Checking {len(samples)} uploaded {category} samples ...")
        seen: set[tuple[str, str]] = set()
        for sample in samples:
            action, wanted = plan_remote_sample(
                str(sample.get("filename", "")),
                sample.get("label"),
                category,
                desired,
                seen,
            )
            sample_id = int(sample["id"])
            if action == "delete":
                delete_ids[category].append(sample_id)
            elif action == "relabel" and wanted is not None:
                relabel_ids[(category, wanted)].append(sample_id)
                present.add((category, Path(str(sample.get("filename", ""))).stem))
            else:
                present.add((category, Path(str(sample.get("filename", ""))).stem))

    for category, ids in sorted(delete_ids.items()):
        print(f"Deleting {len(ids)} {category} samples that are not in CNN-Data.")
        for batch in chunks(ids, 400):
            query = urlencode({"category": category, "ids": json.dumps(batch)})
            result = edge_api_post(api_key, f"/api/{project_id}/raw-data/batch/delete?{query}", {})
            wait_for_job(api_key, project_id, result)

    for (category, label), ids in sorted(relabel_ids.items()):
        print(f"Relabeling {len(ids)} samples: {category} / {label}")
        for batch in chunks(ids, 400):
            query = urlencode({"category": category, "ids": json.dumps(batch)})
            result = edge_api_post(
                api_key,
                f"/api/{project_id}/raw-data/batch/edit-labels?{query}",
                {"label": label},
            )
            wait_for_job(api_key, project_id, result)

    missing: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for key, label in desired.items():
        if key not in present:
            missing[(key[0], label)].append(path_by_key[key])

    missing_count = sum(len(paths) for paths in missing.values())
    print(f"Samples still missing from Edge Impulse: {missing_count}")
    for (category, label), paths in sorted(missing.items()):
        upload_group(api_key, hmac_key, category, label, paths, chunk_size, concurrency)

    print("Edge Impulse sync complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload CNN crops to Edge Impulse.")
    parser.add_argument("--key-name", default=DEFAULT_KEY_NAME, help="Name of the API key variable in .env.")
    parser.add_argument(
        "--hmac-key-name",
        default=None,
        help="Optional HMAC key variable in .env. If omitted, common names are tried.",
    )
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Target Edge Impulse project name.")
    parser.add_argument("--project-id", type=int, default=None, help="Target project ID when the key cannot list projects.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Files per uploader call.")
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel uploads inside one batch.")
    parser.add_argument("--sync", action="store_true", help="Delete, relabel, and upload so Edge Impulse matches CNN-Data.")
    parser.add_argument("--limit", type=int, default=None, help="Optional total sample limit for smoke tests.")
    parser.add_argument("--start-category", default=None, help="Resume at this Edge Impulse category.")
    parser.add_argument("--start-label", default=None, help="Resume at this label within the start category.")
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Samples to skip within the first resumed category/label.",
    )
    parser.add_argument("--repair-labels-only", action="store_true", help="Repair uploaded labels and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Show upload counts without sending data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be at least 1")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.start_offset < 0:
        raise ValueError("--start-offset cannot be negative")

    grouped = read_samples(args.limit)
    print_summary(grouped)

    if args.dry_run:
        print("Dry run complete. No files uploaded.")
        return

    api_key = read_env_value(args.key_name)
    if not api_key:
        raise RuntimeError(f"Missing Edge Impulse API key variable: {args.key_name}")
    if not api_key.startswith("ei_"):
        raise RuntimeError("The Edge Impulse API key must start with 'ei_'. Update .env and retry.")
    hmac_names = (args.hmac_key_name,) if args.hmac_key_name else DEFAULT_HMAC_KEY_NAMES
    hmac_key = next((value for name in hmac_names if (value := read_env_value(name))), None)
    if hmac_key:
        print("Using Edge Impulse HMAC key from .env.")

    api_key, project_id = resolve_project_api_key(api_key, args.project_name, args.project_id)

    if args.repair_labels_only:
        repair_uploaded_labels(api_key, project_id)
        return

    if args.sync:
        sync_dataset(api_key, hmac_key, project_id, grouped, args.chunk_size, args.concurrency)
        return

    start_key = None
    if args.start_category or args.start_label:
        if not args.start_category or not args.start_label:
            raise ValueError("--start-category and --start-label must be used together")
        start_key = (args.start_category, args.start_label)

    for category, label in sorted(grouped):
        if start_key and (category, label) < start_key:
            continue
        paths = grouped[(category, label)]
        if start_key and (category, label) == start_key and args.start_offset:
            print(f"Skipping first {args.start_offset} samples for {category} / {label}.")
            paths = paths[args.start_offset :]
        upload_group(api_key, hmac_key, category, label, paths, args.chunk_size, args.concurrency)

    print("Upload complete.")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
