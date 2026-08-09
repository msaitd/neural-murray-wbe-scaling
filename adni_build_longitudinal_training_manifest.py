from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ALL_VISITS = ROOT / "adni_work" / "adni_all_visits_subject_split.csv"
BASELINE = ROOT / "adni_work" / "adni_baseline_subject_split.csv"
OUTPUT = ROOT / "adni_work" / "adni_longitudinal_training_split.csv"
SUMMARY = ROOT / "adni_work" / "adni_longitudinal_training_summary.json"


def image_number(row: dict[str, str]) -> int:
    match = re.fullmatch(r"I(\d+)", row["Image Data ID"].strip(), re.I)
    return int(match.group(1)) if match else 10**12


def within_visit_rank(row: dict[str, str]) -> tuple:
    description = row["Description"].strip()
    return (
        int(description.upper().startswith("MPR-R")),
        int(description.endswith("_2")),
        -(
            int("GradWarp" in description)
            + int("B1 Correction" in description)
            + int("N3" in description)
        ),
        row.get("Acq Date ISO", ""),
        image_number(row),
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    all_rows = read_rows(ALL_VISITS)
    baseline_rows = read_rows(BASELINE)
    baseline_by_subject = {row["Subject"]: row for row in baseline_rows}
    if len(baseline_by_subject) != 639:
        raise ValueError("Expected 639 unique baseline participants.")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["Subject"], row["Visit"].lower())].append(row)

    selected: list[dict[str, str]] = []
    for subject, baseline in sorted(baseline_by_subject.items()):
        split = baseline["Split"]
        if split == "train":
            subject_visits = sorted(
                visit for candidate, visit in grouped if candidate == subject
            )
            for visit in subject_visits:
                row = dict(min(grouped[(subject, visit)], key=within_visit_rank))
                row["Training Role"] = "longitudinal-train"
                selected.append(row)
        else:
            row = dict(baseline)
            row["Training Role"] = "baseline-evaluation"
            selected.append(row)

    selected.sort(
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[row["Split"]],
            row["Subject"],
            row["Visit"],
        )
    )
    fields = list(dict.fromkeys(key for row in selected for key in row))
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    split_subjects = {
        split: {row["Subject"] for row in selected if row["Split"] == split}
        for split in ("train", "validation", "test")
    }
    overlap = {
        "train_validation": len(
            split_subjects["train"].intersection(split_subjects["validation"])
        ),
        "train_test": len(
            split_subjects["train"].intersection(split_subjects["test"])
        ),
        "validation_test": len(
            split_subjects["validation"].intersection(split_subjects["test"])
        ),
    }
    duplicate_subject_visit = Counter(
        (row["Subject"], row["Visit"]) for row in selected
    )
    duplicate_subject_visit = {
        f"{subject}|{visit}": count
        for (subject, visit), count in duplicate_subject_visit.items()
        if count != 1
    }
    summary = {
        "status": "PASS",
        "rows": len(selected),
        "subjects": len({row["Subject"] for row in selected}),
        "row_counts_by_split": dict(Counter(row["Split"] for row in selected)),
        "subject_counts_by_split": {
            split: len(subjects) for split, subjects in split_subjects.items()
        },
        "training_rows_by_visit": dict(
            Counter(
                row["Visit"]
                for row in selected
                if row["Split"] == "train"
            )
        ),
        "group_rows_by_split": {
            split: dict(
                Counter(
                    row["Group"] for row in selected if row["Split"] == split
                )
            )
            for split in ("train", "validation", "test")
        },
        "subject_overlap": overlap,
        "duplicate_subject_visit": duplicate_subject_visit,
        "output": str(OUTPUT),
    }
    if (
        summary["subjects"] != 639
        or any(overlap.values())
        or duplicate_subject_visit
        or summary["subject_counts_by_split"]
        != {"train": 448, "validation": 96, "test": 95}
    ):
        summary["status"] = "FAIL"
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
