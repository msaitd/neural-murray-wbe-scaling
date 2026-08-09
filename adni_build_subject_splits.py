from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "adni_work" / "adni_manifest.csv"
BASELINE_OUTPUT = ROOT / "adni_work" / "adni_baseline_subject_split.csv"
ALL_VISITS_OUTPUT = ROOT / "adni_work" / "adni_all_visits_subject_split.csv"
SUMMARY_OUTPUT = ROOT / "adni_work" / "adni_split_summary.json"
SEED = 20260727
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CLASS_INDEX = {"AD": 1, "CN": 2, "MCI": 3}


def image_number(row: dict[str, str]) -> int:
    match = re.fullmatch(r"I(\d+)", row["Image Data ID"].strip(), re.I)
    return int(match.group(1)) if match else 10**12


def canonical_rank(row: dict[str, str]) -> tuple:
    visit = row["Visit"].strip().lower()
    visit_priority = {"sc": 0, "bl": 1, "uns1": 2, "m06": 3, "m12": 4}
    description = row["Description"].strip()
    repeat_penalty = int(description.upper().startswith("MPR-R"))
    scaled_two_penalty = int(description.endswith("_2"))
    correction_score = -(
        int("GradWarp" in description)
        + int("B1 Correction" in description)
        + int("N3" in description)
    )
    return (
        visit_priority.get(visit, 99),
        repeat_penalty,
        scaled_two_penalty,
        correction_score,
        row.get("Acq Date ISO", ""),
        image_number(row),
    )


def allocate_split_counts(total: int) -> dict[str, int]:
    exact = {key: total * value for key, value in RATIOS.items()}
    counts = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        RATIOS,
        key=lambda key: (exact[key] - counts[key], RATIOS[key]),
        reverse=True,
    )
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def main() -> None:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["Match Status"] != "matched":
            raise ValueError(
                f"Unmatched image {row['Image Data ID']}: {row['Match Status']}"
            )
        by_subject[row["Subject"].strip()].append(row)

    baseline_rows = []
    group_subjects: dict[str, list[str]] = defaultdict(list)
    for subject, subject_rows in sorted(by_subject.items()):
        groups = {row["Group"].strip() for row in subject_rows}
        if len(groups) != 1:
            raise ValueError(f"Subject {subject} has multiple groups: {groups}")
        selected = min(subject_rows, key=canonical_rank)
        selected = dict(selected)
        selected["Selection Rank"] = "|".join(map(str, canonical_rank(selected)))
        selected["Subject Image Count"] = str(len(subject_rows))
        selected["Class Index"] = str(CLASS_INDEX[selected["Group"].strip()])
        baseline_rows.append(selected)
        group_subjects[selected["Group"].strip()].append(subject)

    rng = random.Random(SEED)
    subject_split: dict[str, str] = {}
    split_group_counts: Counter[tuple[str, str]] = Counter()
    for group, subjects in sorted(group_subjects.items()):
        ordered = sorted(subjects)
        rng.shuffle(ordered)
        counts = allocate_split_counts(len(ordered))
        cursor = 0
        for split in ("train", "validation", "test"):
            for subject in ordered[cursor : cursor + counts[split]]:
                subject_split[subject] = split
                split_group_counts[(split, group)] += 1
            cursor += counts[split]
        if cursor != len(ordered):
            raise AssertionError("Split allocation did not consume every subject.")

    for row in baseline_rows:
        row["Split"] = subject_split[row["Subject"].strip()]

    baseline_fields = list(baseline_rows[0]) + ["Split"]
    baseline_fields = list(dict.fromkeys(baseline_fields))
    with BASELINE_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline_fields)
        writer.writeheader()
        writer.writerows(baseline_rows)

    all_visit_rows = []
    for row in rows:
        enriched = dict(row)
        enriched["Split"] = subject_split[row["Subject"].strip()]
        enriched["Class Index"] = str(CLASS_INDEX[row["Group"].strip()])
        all_visit_rows.append(enriched)

    all_visit_fields = list(all_visit_rows[0])
    with ALL_VISITS_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_visit_fields)
        writer.writeheader()
        writer.writerows(all_visit_rows)

    overlap_checks = {}
    split_sets = {
        split: {
            subject for subject, assigned in subject_split.items() if assigned == split
        }
        for split in RATIOS
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap_checks[f"{left}_{right}"] = len(
            split_sets[left].intersection(split_sets[right])
        )

    selected_visit_counts = Counter(row["Visit"] for row in baseline_rows)
    selected_description_counts = Counter(
        row["Description"] for row in baseline_rows
    )
    summary = {
        "seed": SEED,
        "ratios": RATIOS,
        "class_index": CLASS_INDEX,
        "subjects": len(by_subject),
        "baseline_rows": len(baseline_rows),
        "all_visit_rows": len(all_visit_rows),
        "split_subject_counts": {
            split: len(subjects) for split, subjects in split_sets.items()
        },
        "split_group_counts": {
            split: {
                group: split_group_counts[(split, group)]
                for group in sorted(group_subjects)
            }
            for split in RATIOS
        },
        "selected_visit_counts": dict(selected_visit_counts),
        "selected_description_counts": dict(selected_description_counts),
        "subject_overlap_checks": overlap_checks,
        "outputs": {
            "baseline": str(BASELINE_OUTPUT),
            "all_visits": str(ALL_VISITS_OUTPUT),
        },
    }
    SUMMARY_OUTPUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
