from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "adni_work" / "adni_3d" / "adni_3d_manifest.csv"
DEFAULT_OUTPUT = ROOT / "adni_work" / "adni_3d_cv"
CLASS_INDEX = {"AD": "1", "CN": "2", "MCI": "3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic participant-stratified 5-fold manifests."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 639 or len({row["subject"] for row in rows}) != 639:
        raise ValueError("Expected exactly one 3-D row for each of 639 participants.")

    by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["group"] not in CLASS_INDEX:
            raise ValueError(f"Unexpected group: {row['group']}")
        by_group[row["group"]].append(row)

    assignment: dict[str, int] = {}
    for group in sorted(by_group):
        group_rows = sorted(by_group[group], key=lambda row: row["subject"])
        # Use a stable class-specific offset rather than Python's randomized hash.
        rng = random.Random(args.seed + {"AD": 11, "CN": 23, "MCI": 37}[group])
        rng.shuffle(group_rows)
        for index, row in enumerate(group_rows):
            assignment[row["subject"]] = index % 5 + 1

    args.output.mkdir(parents=True, exist_ok=True)
    assignment_path = args.output / "adni_3d_5fold_assignments.csv"
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["subject", "group", "classIndex", "outerFold"]
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["subject"]):
            writer.writerow(
                {
                    "subject": row["subject"],
                    "group": row["group"],
                    "classIndex": CLASS_INDEX[row["group"]],
                    "outerFold": assignment[row["subject"]],
                }
            )

    fold_summaries: dict[str, dict[str, object]] = {}
    for outer_fold in range(1, 6):
        validation_fold = outer_fold % 5 + 1
        fold_root = args.output / f"fold_{outer_fold}"
        fold_root.mkdir(exist_ok=True)
        output_rows: list[dict[str, str]] = []
        for row in rows:
            fold = assignment[row["subject"]]
            updated = dict(row)
            if fold == outer_fold:
                updated["split"] = "test"
            elif fold == validation_fold:
                updated["split"] = "validation"
            else:
                updated["split"] = "train"
            output_rows.append(updated)

        manifest_path = fold_root / "adni_3d_manifest.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(output_rows)

        split_counts = Counter(row["split"] for row in output_rows)
        test_groups = Counter(
            row["group"] for row in output_rows if row["split"] == "test"
        )
        validation_groups = Counter(
            row["group"] for row in output_rows if row["split"] == "validation"
        )
        train_groups = Counter(
            row["group"] for row in output_rows if row["split"] == "train"
        )
        fold_summaries[str(outer_fold)] = {
            "validation_fold": validation_fold,
            "split_counts": dict(split_counts),
            "train_groups": dict(train_groups),
            "validation_groups": dict(validation_groups),
            "test_groups": dict(test_groups),
            "manifest": str(manifest_path),
        }

    summary = {
        "status": "PASS",
        "seed": args.seed,
        "participants": len(rows),
        "folds": 5,
        "assignment": str(assignment_path),
        "fold_summaries": fold_summaries,
        "note": (
            "Each outer test fold is used once. The next cyclic fold is validation; "
            "the remaining three folds are training."
        ),
    }
    summary_path = args.output / "adni_3d_5fold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
