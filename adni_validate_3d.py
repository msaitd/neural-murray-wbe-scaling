from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent / "adni_work" / "adni_3d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate materialized ADNI 3-D data.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.root / "adni_3d_manifest.csv"
    output = args.output or args.root.parent / "adni_3d_validation.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "subject",
        "imageDataId",
        "group",
        "classIndex",
        "split",
        "niftiPath",
        "matPath",
    }
    missing_columns = sorted(required.difference(rows[0] if rows else {}))

    subjects = [row["subject"] for row in rows]
    paths = [Path(row["matPath"]) for row in rows]
    missing_files = [str(path) for path in paths if not path.is_file()]
    empty_files = [str(path) for path in paths if path.is_file() and path.stat().st_size == 0]
    duplicate_subjects = {
        key: value for key, value in Counter(subjects).items() if value != 1
    }
    duplicate_paths = len(paths) - len(set(paths))

    split_subjects: dict[str, set[str]] = {}
    for row in rows:
        split_subjects.setdefault(row["split"], set()).add(row["subject"])
    split_names = sorted(split_subjects)
    overlap: dict[str, int] = {}
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap[f"{left}_{right}"] = len(
                split_subjects[left].intersection(split_subjects[right])
            )

    split_counts = Counter(row["split"] for row in rows)
    group_counts = Counter(row["group"] for row in rows)
    class_mapping: dict[str, set[str]] = {}
    for row in rows:
        class_mapping.setdefault(row["group"], set()).add(row["classIndex"])
    invalid_class_mapping = {
        group: sorted(values)
        for group, values in class_mapping.items()
        if len(values) != 1
    }

    status = "PASS"
    if (
        missing_columns
        or len(rows) != 639
        or len(set(subjects)) != 639
        or missing_files
        or empty_files
        or duplicate_subjects
        or duplicate_paths
        or any(overlap.values())
        or invalid_class_mapping
    ):
        status = "FAIL"

    result = {
        "status": status,
        "manifest": str(manifest),
        "rows": len(rows),
        "subjects": len(set(subjects)),
        "split_subject_counts": dict(split_counts),
        "group_subject_counts": dict(group_counts),
        "missing_columns": missing_columns,
        "missing_files": len(missing_files),
        "empty_files": len(empty_files),
        "duplicate_subjects": duplicate_subjects,
        "duplicate_paths": duplicate_paths,
        "subject_overlap": overlap,
        "invalid_class_mapping": invalid_class_mapping,
    }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
