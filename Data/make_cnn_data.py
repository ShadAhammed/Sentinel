"""
make_cnn_data.py

Copy a smaller CNN set from Data/CNN into Data/CNN-Data.

Rules for the working CNN set:
    - Drop civilian road users, the generic Vehicle class, civilian, civilian_vehicle, and weapon.
    - Merge military_tank, Tank, and military_truck into military_vehicle.
    - Merge military_artillery and the KIIT spelling Artilary into Artillery.
    - Merge Soldier_KIIT-MiTA and soldier_military_object_dataset into Soldier.
    - Training images (train + validation together) stay at or below 1000 per label.
      Validation images are kept first, then train fills the rest of the 1000.
      A fresh copy from Data/CNN also caps validation at 100 images per label.
    - Test stays at or below 50 images per label.

If a label already has fewer images than the cap, all of them are kept.
The same rules can be applied to the CNN-Data folder that is already on disk.
"""

from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


# Paths come from this file, not from a hardcoded drive letter.
DATA_DIR = Path(__file__).resolve().parent
CNN_DIR = DATA_DIR / "CNN"
OUT_DIR = DATA_DIR / "CNN-Data"
SRC_CSV = CNN_DIR / "labels.csv"
OUT_CSV = OUT_DIR / "labels.csv"

# Same random seed every run so the subset can be reproduced.
RANDOM_SEED = 42

# Training cap is train and validation combined. Test is a separate cap.
TRAINING_MAX = 1000
TEST_MAX = 50

# Exact label match, compared without case. Do not match these as substrings,
# so military_vehicle stays in the set.
DROP_LABELS = {
    "car",
    "vehicle",
    "van",
    "truck",
    "bus",
    "motor",
    "bicycle",
    "tricycle",
    "awning-tricycle",
    "people",
    "pedestrian",
    "civilian",
    "civilian_vehicle",
    "weapon",
}

# Source label -> folder and label name to keep.
LABEL_MERGE = {
    "military_tank": "military_vehicle",
    "Tank": "military_vehicle",
    "military_truck": "military_vehicle",
    "military_artillery": "Artillery",
    "Artilary": "Artillery",
    "Soldier_KIIT-MiTA": "Soldier",
    "soldier_military_object_dataset": "Soldier",
    # The military_object_dataset YAML spells this class "soldier".
    "soldier": "Soldier",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def mapped_label(label: str) -> str | None:
    """Return the label to keep, or None when the image should be removed."""
    if label.lower() in DROP_LABELS:
        return None
    return LABEL_MERGE.get(label, label)


def prepare_row(row: dict[str, str]) -> dict[str, str] | None:
    """Copy one CSV row and point it at the merged label folder."""
    label = mapped_label(row["label"])
    if label is None:
        return None

    split = row["split"]
    filename = Path(row["crop_file"]).name
    prepared = dict(row)
    prepared["label"] = label
    prepared["source_crop"] = row["crop_file"]
    # Path builds a platform path. CSV writing keeps that slash style.
    prepared["crop_file"] = str(Path(split) / label / filename)
    return prepared


def take_rows(rows: list[dict[str, str]], cap: int, rng: random.Random) -> list[dict[str, str]]:
    """Keep every row under the cap. Otherwise draw `cap` rows from a sorted list."""
    ordered = sorted(rows, key=lambda row: row["source_crop"])
    if cap <= 0 or not ordered:
        return []
    if len(ordered) <= cap:
        return ordered
    return rng.sample(ordered, cap)


def select_rows(
    rows: list[dict[str, str]],
    rng: random.Random,
    validation_max: int | None = None,
) -> list[dict[str, str]]:
    """Apply drops, merges, the 1000 training cap, and the 50 test cap.

    validation_max limits the validation split before train fills the rest of
    the 1000. The reshape of the current CNN-Data folder leaves this unset,
    because that validation split is already small. A fresh copy from Data/CNN
    passes 100 so validation cannot use the whole training budget.
    """
    grouped: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"train": [], "validation": [], "test": []}
    )
    for row in rows:
        prepared = prepare_row(row)
        if prepared is None:
            continue
        split = prepared["split"]
        if split not in grouped[prepared["label"]]:
            raise ValueError(f"Unsupported split: {split}")
        grouped[prepared["label"]][split].append(prepared)

    chosen: list[dict[str, str]] = []
    for label in sorted(grouped):
        splits = grouped[label]
        # Validation is the smaller holdout, so it uses the training budget first.
        val_cap = TRAINING_MAX if validation_max is None else min(validation_max, TRAINING_MAX)
        validation = take_rows(splits["validation"], val_cap, rng)
        train_room = TRAINING_MAX - len(validation)
        train = take_rows(splits["train"], train_room, rng)
        test = take_rows(splits["test"], TEST_MAX, rng)
        chosen.extend(train)
        chosen.extend(validation)
        chosen.extend(test)
    return chosen


def safe_dest(crop_file: str) -> Path | None:
    """
    Build the CNN-Data path for a crop.
    Refuse to write outside CNN-Data (blocks .. in crop_file).
    """
    dest = (OUT_DIR / crop_file).resolve()
    try:
        dest.relative_to(OUT_DIR.resolve())
    except ValueError:
        print(f"Skip unsafe crop path: {crop_file}")
        return None
    return dest


def remove_empty_dirs(root: Path) -> None:
    """Remove label folders that no longer contain images."""
    for folder in sorted(root.rglob("*"), reverse=True):
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def reshape_existing() -> None:
    """Apply the CNN-Data rules to the folder that is already on disk."""
    if not OUT_CSV.is_file():
        print(f"Missing {OUT_CSV}")
        return

    print(f"Reading {OUT_CSV} ...")
    with OUT_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    chosen = select_rows(rows, random.Random(RANDOM_SEED))
    keep_paths: set[Path] = set()
    for row in chosen:
        src = safe_dest(row["source_crop"])
        dest = safe_dest(row["crop_file"])
        if src is None or dest is None:
            raise RuntimeError("Refusing to reshape because a crop path escapes CNN-Data.")
        if dest.exists() and src != dest:
            raise RuntimeError(f"Refusing to overwrite an existing crop: {dest.name}")
        keep_paths.add(dest)

    print(f"Keeping {len(chosen)} images. Moving merged labels ...")
    for index, row in enumerate(chosen, start=1):
        src = safe_dest(row["source_crop"])
        dest = safe_dest(row["crop_file"])
        if src is None or dest is None or not src.is_file():
            raise RuntimeError(f"Missing crop listed in labels.csv: {row['source_crop']}")
        if src != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dest)
        if index % 2000 == 0:
            print(f"  moved {index}/{len(chosen)}")

    # Delete images that were dropped, merged away, or cut by the caps.
    removed = 0
    for split in ("train", "validation", "test"):
        split_dir = OUT_DIR / split
        if not split_dir.is_dir():
            continue
        for path in split_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.resolve() not in keep_paths:
                path.unlink()
                removed += 1
        remove_empty_dirs(split_dir)

    chosen.sort(key=lambda row: (row["split"], row["label"], row["crop_file"]))
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)

    print(f"Done. Kept {len(chosen)} images. Removed {removed} images.")
    print(f"Label file: {OUT_CSV}")


def main() -> None:
    """Build CNN-Data by copying a capped subset out of Data/CNN."""
    if not SRC_CSV.is_file():
        print(f"Missing {SRC_CSV}. Run localize.py first.")
        return

    print(f"Reading {SRC_CSV} ...")
    with SRC_CSV.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Keep the fresh validation split small, then let train fill the 1000 cap.
    chosen = select_rows(rows, random.Random(RANDOM_SEED), validation_max=100)

    # Create the three split folders even if a split were empty.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "train").mkdir(exist_ok=True)
    (OUT_DIR / "test").mkdir(exist_ok=True)
    (OUT_DIR / "validation").mkdir(exist_ok=True)

    print(f"Copying {len(chosen)} crops into {OUT_DIR} ...")
    copied: list[dict[str, str]] = []
    missing = 0

    for i, row in enumerate(chosen, start=1):
        src = CNN_DIR / row["source_crop"]
        dest = safe_dest(row["crop_file"])
        if dest is None:
            continue
        if not src.is_file():
            missing += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(row)

        # Progress so a long copy does not look stuck.
        if i % 5000 == 0:
            print(f"  copied {i}/{len(chosen)}")

    # Sort the new label file so it is easy to scan.
    copied.sort(key=lambda row: (row["split"], row["label"], row["crop_file"]))

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(copied)

    print(f"Done. Copied {len(copied)} images.")
    print(f"Missing source files skipped: {missing}")
    print(f"Label file: {OUT_CSV}")


if __name__ == "__main__":
    main()
