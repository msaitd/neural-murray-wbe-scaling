from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "adni_work"
SOURCE = ROOT / "adni_longitudinal_training_split.csv"
MANIFEST_25D = ROOT / "adni_25d_longitudinal" / "adni_25d_manifest.csv"
MANIFEST_3D = ROOT / "adni_3d_longitudinal" / "adni_3d_manifest.csv"
OUTPUT = ROOT / "adni_longitudinal_validation.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_overlap(rows: list[dict[str, str]]) -> dict[str, int]:
    sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sets[row["split"]].add(row["subject"])
    return {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }


def main() -> int:
    source = read_rows(SOURCE)
    rows_25d = read_rows(MANIFEST_25D)
    rows_3d = read_rows(MANIFEST_3D)
    expected_images = len(source) * 4

    counts_25d = Counter(
        (row["subject"], row["imageDataId"]) for row in rows_25d
    )
    invalid_25d_multiplicity = {
        f"{subject}|{image_id}": count
        for (subject, image_id), count in counts_25d.items()
        if count != 4
    }
    counts_3d = Counter(
        (row["subject"], row["imageDataId"]) for row in rows_3d
    )
    invalid_3d_multiplicity = {
        f"{subject}|{image_id}": count
        for (subject, image_id), count in counts_3d.items()
        if count != 1
    }

    paths_25d = [Path(row["pngPath"]) for row in rows_25d]
    paths_3d = [Path(row["matPath"]) for row in rows_3d]
    missing_25d = sum(not path.is_file() for path in paths_25d)
    missing_3d = sum(not path.is_file() for path in paths_3d)
    empty_25d = sum(path.is_file() and path.stat().st_size == 0 for path in paths_25d)
    empty_3d = sum(path.is_file() and path.stat().st_size == 0 for path in paths_3d)
    overlap_25d = split_overlap(rows_25d)
    overlap_3d = split_overlap(rows_3d)

    result = {
        "status": "PASS",
        "source_rows": len(source),
        "source_subjects": len({row["Subject"] for row in source}),
        "expected_25d_images": expected_images,
        "actual_25d_images": len(rows_25d),
        "actual_3d_volumes": len(rows_3d),
        "split_25d_images": dict(Counter(row["split"] for row in rows_25d)),
        "split_3d_volumes": dict(Counter(row["split"] for row in rows_3d)),
        "invalid_25d_multiplicity": invalid_25d_multiplicity,
        "invalid_3d_multiplicity": invalid_3d_multiplicity,
        "duplicate_25d_paths": len(paths_25d) - len(set(paths_25d)),
        "duplicate_3d_paths": len(paths_3d) - len(set(paths_3d)),
        "missing_25d_files": missing_25d,
        "missing_3d_files": missing_3d,
        "empty_25d_files": empty_25d,
        "empty_3d_files": empty_3d,
        "subject_overlap_25d": overlap_25d,
        "subject_overlap_3d": overlap_3d,
    }
    if (
        len(source) != 1535
        or result["source_subjects"] != 639
        or len(rows_25d) != expected_images
        or len(rows_3d) != len(source)
        or invalid_25d_multiplicity
        or invalid_3d_multiplicity
        or result["duplicate_25d_paths"]
        or result["duplicate_3d_paths"]
        or missing_25d
        or missing_3d
        or empty_25d
        or empty_3d
        or any(overlap_25d.values())
        or any(overlap_3d.values())
    ):
        result["status"] = "FAIL"
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
