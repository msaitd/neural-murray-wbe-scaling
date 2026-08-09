from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "adni_work"
ALL_VISITS = WORK / "adni_all_visits_subject_split.csv"
PREPARED_LONGITUDINAL = (
    WORK / "adni_3d_longitudinal" / "adni_3d_manifest.csv"
)
OLD_CV_ROOT = WORK / "adni_3d_cv"
CV_ROOT = WORK / "adni_3d_cv_v2"
SELECTED_SOURCE = CV_ROOT / "selected_all_visits_source.csv"
MISSING_SPLIT = CV_ROOT / "missing_3d_split.csv"
MISSING_PREPARED = (
    CV_ROOT / "missing_volumes" / "adni_3d_manifest.csv"
)
PLAN_SUMMARY = CV_ROOT / "manifest_plan_summary.json"
FINAL_SUMMARY = CV_ROOT / "manifest_validation.json"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(
    path: Path, rows: list[dict[str, str]], fields: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def image_number(row: dict[str, str]) -> int:
    match = re.fullmatch(
        r"I(\d+)", row["Image Data ID"].strip(), re.IGNORECASE
    )
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


def selected_visits() -> list[dict[str, str]]:
    rows = read_rows(ALL_VISITS)
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(
        list
    )
    for row in rows:
        grouped[(row["Subject"], row["Visit"].lower())].append(row)
    selected = [
        dict(min(group, key=within_visit_rank))
        for group in grouped.values()
    ]
    selected.sort(
        key=lambda row: (
            row["Subject"],
            row["Visit"].lower(),
            image_number(row),
        )
    )
    subjects = {row["Subject"] for row in selected}
    counts = Counter(row["Subject"] for row in selected)
    if (
        len(subjects) != 639
        or len(selected) != 1917
        or set(counts.values()) != {3}
    ):
        raise ValueError(
            "Expected exactly three canonical visits for 639 participants."
        )
    return selected


def prepared_map(
    paths: list[Path],
) -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for row in read_rows(path):
            key = (row["subject"], row["imageDataId"])
            if key in result:
                raise ValueError(f"Duplicate prepared volume: {key}")
            if not Path(row["matPath"]).is_file():
                raise FileNotFoundError(row["matPath"])
            result[key] = row
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def plan() -> None:
    CV_ROOT.mkdir(parents=True, exist_ok=True)
    selected = selected_visits()
    fields = list(
        dict.fromkeys(key for row in selected for key in row.keys())
    )
    write_rows(SELECTED_SOURCE, selected, fields)

    existing = prepared_map([PREPARED_LONGITUDINAL])
    missing = [
        dict(row)
        for row in selected
        if (row["Subject"], row["Image Data ID"]) not in existing
    ]
    for row in missing:
        row["Split"] = "pool"
        row["Class Index"] = {
            "AD": "1",
            "CN": "2",
            "MCI": "3",
        }[row["Group"]]
    missing_fields = list(
        dict.fromkeys(key for row in missing for key in row.keys())
    )
    write_rows(MISSING_SPLIT, missing, missing_fields)

    missing_subjects = {row["Subject"] for row in missing}
    result = {
        "status": "PASS",
        "selected_rows": len(selected),
        "selected_subjects": len(
            {row["Subject"] for row in selected}
        ),
        "existing_prepared_rows": len(existing),
        "missing_prepared_rows": len(missing),
        "missing_subjects": len(missing_subjects),
        "missing_rows_per_subject": dict(
            sorted(Counter(row["Subject"] for row in missing).items())
        ),
        "selected_source": str(SELECTED_SOURCE),
        "selected_source_sha256": sha256(SELECTED_SOURCE),
        "missing_split": str(MISSING_SPLIT),
        "missing_split_sha256": sha256(MISSING_SPLIT),
        "missing_output_root": str(CV_ROOT / "missing_volumes"),
    }
    if (
        len(existing) != 1535
        or len(missing) != 382
        or len(missing_subjects) != 191
        or set(result["missing_rows_per_subject"].values()) != {2}
    ):
        result["status"] = "FAIL"
    PLAN_SUMMARY.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


def make_train_row(
    source: dict[str, str],
    prepared: dict[str, str],
) -> dict[str, str]:
    return {
        "subject": source["Subject"],
        "imageDataId": source["Image Data ID"],
        "group": source["Group"],
        "classIndex": source["Class Index"],
        "split": "train",
        "visit": source["Visit"],
        "niftiPath": source["NIfTI Path"],
        "matPath": prepared["matPath"],
        "intensityLower": prepared.get("intensityLower", ""),
        "intensityUpper": prepared.get("intensityUpper", ""),
        "nonzeroFraction": prepared.get("nonzeroFraction", ""),
        "originalSize": prepared.get("originalSize", ""),
        "reorientedSize": prepared.get("reorientedSize", ""),
        "croppedSize": prepared.get("croppedSize", ""),
        "sourceOriginalSplit": source["Split"],
    }


def make_evaluation_row(
    baseline: dict[str, str],
    source: dict[str, str],
) -> dict[str, str]:
    return {
        "subject": baseline["subject"],
        "imageDataId": baseline["imageDataId"],
        "group": baseline["group"],
        "classIndex": baseline["classIndex"],
        "split": baseline["split"],
        "visit": source["Visit"],
        "niftiPath": baseline["niftiPath"],
        "matPath": baseline["matPath"],
        "intensityLower": baseline.get("intensityLower", ""),
        "intensityUpper": baseline.get("intensityUpper", ""),
        "nonzeroFraction": baseline.get("nonzeroFraction", ""),
        "originalSize": baseline.get("originalSize", ""),
        "reorientedSize": baseline.get("reorientedSize", ""),
        "croppedSize": baseline.get("croppedSize", ""),
        "sourceOriginalSplit": source["Split"],
    }


def subject_group_counts(
    rows: list[dict[str, str]], split: str
) -> dict[str, int]:
    subject_groups: dict[str, str] = {}
    for row in rows:
        if row["split"] != split:
            continue
        subject = row["subject"]
        group = row["group"]
        if subject in subject_groups and subject_groups[subject] != group:
            raise ValueError(
                f"Participant {subject} changes group within fold."
            )
        subject_groups[subject] = group
    return dict(Counter(subject_groups.values()))


def finalize() -> None:
    if not MISSING_PREPARED.is_file():
        raise FileNotFoundError(
            "Prepare the missing volumes before finalization: "
            f"{MISSING_PREPARED}"
        )
    selected = selected_visits()
    selected_by_key = {
        (row["Subject"], row["Image Data ID"]): row
        for row in selected
    }
    selected_by_subject: dict[str, list[dict[str, str]]] = defaultdict(
        list
    )
    for row in selected:
        selected_by_subject[row["Subject"]].append(row)
    prepared = prepared_map(
        [PREPARED_LONGITUDINAL, MISSING_PREPARED]
    )
    selected_keys = set(selected_by_key)
    if set(prepared) != selected_keys:
        missing = sorted(selected_keys - set(prepared))
        extra = sorted(set(prepared) - selected_keys)
        raise ValueError(
            f"Prepared pool mismatch; missing={missing[:3]} "
            f"extra={extra[:3]}"
        )

    output_fields = [
        "subject",
        "imageDataId",
        "group",
        "classIndex",
        "split",
        "visit",
        "niftiPath",
        "matPath",
        "intensityLower",
        "intensityUpper",
        "nonzeroFraction",
        "originalSize",
        "reorientedSize",
        "croppedSize",
        "sourceOriginalSplit",
    ]
    reports = []
    outer_test_counter: Counter[str] = Counter()
    validation_counter: Counter[str] = Counter()
    training_counter: Counter[str] = Counter()
    for fold in range(1, 6):
        old_manifest = (
            OLD_CV_ROOT / f"fold_{fold}" / "adni_3d_manifest.csv"
        )
        baseline_rows = read_rows(old_manifest)
        baseline_by_subject = {
            row["subject"]: row for row in baseline_rows
        }
        if len(baseline_by_subject) != 639:
            raise ValueError(
                f"Fold {fold} does not contain 639 baseline subjects."
            )
        rows: list[dict[str, str]] = []
        for subject in sorted(baseline_by_subject):
            baseline = baseline_by_subject[subject]
            split = baseline["split"]
            if split == "train":
                visits = selected_by_subject[subject]
                if len(visits) != 3:
                    raise ValueError(
                        f"{subject} does not have three visits."
                    )
                for source in visits:
                    key = (subject, source["Image Data ID"])
                    rows.append(make_train_row(source, prepared[key]))
                training_counter[subject] += 1
            else:
                key = (subject, baseline["imageDataId"])
                source = selected_by_key.get(key)
                if source is None:
                    raise ValueError(
                        f"Fold baseline is not canonical: {key}"
                    )
                rows.append(make_evaluation_row(baseline, source))
                if split == "test":
                    outer_test_counter[subject] += 1
                elif split == "validation":
                    validation_counter[subject] += 1
                else:
                    raise ValueError(f"Unknown split: {split}")

        rows.sort(
            key=lambda row: (
                {"train": 0, "validation": 1, "test": 2}[
                    row["split"]
                ],
                row["subject"],
                row["visit"],
            )
        )
        output = (
            CV_ROOT / f"fold_{fold}" / "adni_3d_manifest.csv"
        )
        write_rows(output, rows, output_fields)
        split_rows = Counter(row["split"] for row in rows)
        split_subjects = {
            split: {
                row["subject"]
                for row in rows
                if row["split"] == split
            }
            for split in ("train", "validation", "test")
        }
        overlaps = {
            "train_validation": len(
                split_subjects["train"]
                & split_subjects["validation"]
            ),
            "train_test": len(
                split_subjects["train"] & split_subjects["test"]
            ),
            "validation_test": len(
                split_subjects["validation"]
                & split_subjects["test"]
            ),
        }
        training_visits = Counter(
            row["subject"]
            for row in rows
            if row["split"] == "train"
        )
        missing_files = sum(
            not Path(row["matPath"]).is_file() for row in rows
        )
        report = {
            "fold": fold,
            "manifest": str(output),
            "sha256": sha256(output),
            "row_counts": dict(split_rows),
            "subject_counts": {
                split: len(subjects)
                for split, subjects in split_subjects.items()
            },
            "class_subject_counts": {
                split: subject_group_counts(rows, split)
                for split in ("train", "validation", "test")
            },
            "training_visit_multiplicity": dict(
                Counter(training_visits.values())
            ),
            "subject_overlap": overlaps,
            "missing_files": missing_files,
        }
        expected_train_rows = 3 * report["subject_counts"]["train"]
        evaluation_visits = {
            row["visit"].lower()
            for row in rows
            if row["split"] in {"validation", "test"}
        }
        if (
            report["row_counts"]["train"] != expected_train_rows
            or report["row_counts"]["validation"]
            != report["subject_counts"]["validation"]
            or report["row_counts"]["test"]
            != report["subject_counts"]["test"]
            or report["training_visit_multiplicity"]
            != {3: report["subject_counts"]["train"]}
            or evaluation_visits != {"sc"}
            or any(
                set(report["class_subject_counts"][split])
                != {"AD", "CN", "MCI"}
                for split in ("train", "validation", "test")
            )
            or any(overlaps.values())
            or missing_files
        ):
            raise ValueError(f"Fold {fold} validation failed.")
        reports.append(report)

    all_subjects = set(selected_by_subject)
    repeated_counts_ok = all(
        outer_test_counter[subject] == 1
        and validation_counter[subject] == 1
        and training_counter[subject] == 3
        for subject in all_subjects
    )
    result = {
        "status": "PASS" if repeated_counts_ok else "FAIL",
        "subjects": len(all_subjects),
        "canonical_visits": len(selected),
        "prepared_volumes": len(prepared),
        "folds": reports,
        "outer_test_multiplicity": dict(
            Counter(outer_test_counter.values())
        ),
        "validation_multiplicity": dict(
            Counter(validation_counter.values())
        ),
        "training_multiplicity": dict(
            Counter(training_counter.values())
        ),
        "outer_test_predictions_evaluated": False,
    }
    FINAL_SUMMARY.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("plan", "finalize"), required=True
    )
    arguments = parser.parse_args()
    if arguments.stage == "plan":
        plan()
    else:
        finalize()


if __name__ == "__main__":
    main()
