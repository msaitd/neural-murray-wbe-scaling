from __future__ import annotations

import csv
import json
import os
import re
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match an authorized ADNI image-metadata export to a local "
            "NIfTI tree without redistributing either input."
        )
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "adni_work"
    )
    return parser.parse_args()


def parse_date(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            pass
    return value


def image_id_from_path(path: Path) -> str | None:
    match = re.search(r"(?:^|_)I(\d+)(?:\.nii(?:\.gz)?)$", path.name, re.I)
    if match:
        return f"I{match.group(1)}"
    for part in reversed(path.parts):
        if re.fullmatch(r"I\d+", part, re.I):
            return part.upper()
    return None


def main() -> None:
    args = parse_args()
    image_root = args.image_root.resolve()
    csv_path = args.metadata_csv.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "adni_manifest.csv"
    subject_path = output_dir / "adni_subject_summary.csv"
    summary_path = output_dir / "adni_inventory_summary.json"

    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, list[str]] = defaultdict(list)
    extension_counts: Counter[str] = Counter()
    total_files = 0
    for root, _, files in os.walk(image_root):
        for file_name in files:
            total_files += 1
            path = Path(root) / file_name
            suffix = ".nii.gz" if file_name.lower().endswith(".nii.gz") else path.suffix.lower()
            extension_counts[suffix or "<none>"] += 1
            if suffix in {".nii", ".nii.gz"}:
                image_id = image_id_from_path(path)
                if image_id:
                    file_map[image_id].append(str(path))

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "Image Data ID",
        "Subject",
        "Group",
        "Visit",
        "Modality",
        "Description",
        "Acq Date",
        "Format",
    }
    missing_fields = sorted(required.difference(source_fields))
    if missing_fields:
        raise ValueError(f"Missing required CSV fields: {missing_fields}")

    id_counts = Counter(row["Image Data ID"].strip() for row in rows)
    subjects: dict[str, list[dict[str, str]]] = defaultdict(list)
    manifest_rows: list[dict[str, str | int]] = []
    matched_ids: set[str] = set()

    for source_row_number, row in enumerate(rows, start=2):
        image_id = row["Image Data ID"].strip().upper()
        subject = row["Subject"].strip()
        paths = sorted(file_map.get(image_id, []))
        if paths:
            matched_ids.add(image_id)
        output_row: dict[str, str | int] = dict(row)
        output_row.update(
            {
                "Source Row": source_row_number,
                "Acq Date ISO": parse_date(row["Acq Date"]),
                "NIfTI Match Count": len(paths),
                "NIfTI Path": paths[0] if len(paths) == 1 else "",
                "Match Status": (
                    "matched"
                    if len(paths) == 1
                    else "missing"
                    if len(paths) == 0
                    else "ambiguous"
                ),
            }
        )
        manifest_rows.append(output_row)
        subjects[subject].append(output_row)

    manifest_fields = source_fields + [
        "Source Row",
        "Acq Date ISO",
        "NIfTI Match Count",
        "NIfTI Path",
        "Match Status",
    ]
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    subject_rows: list[dict[str, str | int]] = []
    for subject, subject_visits in sorted(subjects.items()):
        groups = sorted({str(row["Group"]).strip() for row in subject_visits})
        visits = sorted({str(row["Visit"]).strip() for row in subject_visits})
        dates = sorted(
            str(row["Acq Date ISO"])
            for row in subject_visits
            if str(row["Acq Date ISO"])
        )
        subject_rows.append(
            {
                "Subject": subject,
                "Image Rows": len(subject_visits),
                "Matched Rows": sum(
                    str(row["Match Status"]) == "matched" for row in subject_visits
                ),
                "Groups": "|".join(groups),
                "Group Count": len(groups),
                "Visits": "|".join(visits),
                "First Acquisition": dates[0] if dates else "",
                "Last Acquisition": dates[-1] if dates else "",
            }
        )

    subject_fields = [
        "Subject",
        "Image Rows",
        "Matched Rows",
        "Groups",
        "Group Count",
        "Visits",
        "First Acquisition",
        "Last Acquisition",
    ]
    with subject_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=subject_fields)
        writer.writeheader()
        writer.writerows(subject_rows)

    csv_ids = set(id_counts)
    file_ids = set(file_map)
    status_counts = Counter(str(row["Match Status"]) for row in manifest_rows)
    group_row_counts = Counter(row["Group"].strip() for row in rows)
    visit_row_counts = Counter(row["Visit"].strip() for row in rows)
    description_counts = Counter(row["Description"].strip() for row in rows)
    modality_counts = Counter(row["Modality"].strip() for row in rows)
    format_counts = Counter(row["Format"].strip() for row in rows)
    type_counts = Counter(row["Type"].strip() for row in rows)

    subjects_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        subjects_by_group[row["Group"].strip()].add(row["Subject"].strip())

    summary = {
        "source": {
            "image_root": str(image_root),
            "csv_path": str(csv_path),
            "csv_rows": len(rows),
            "csv_columns": len(source_fields),
            "csv_fields": source_fields,
        },
        "filesystem": {
            "total_files": total_files,
            "extension_counts": dict(extension_counts.most_common()),
            "nifti_ids": len(file_ids),
        },
        "matching": {
            "status_counts": dict(status_counts),
            "duplicate_csv_image_ids": {
                key: value for key, value in id_counts.items() if value > 1
            },
            "csv_ids_without_nifti": len(csv_ids - file_ids),
            "nifti_ids_without_csv": len(file_ids - csv_ids),
            "ambiguous_nifti_ids": sum(
                1 for paths in file_map.values() if len(paths) > 1
            ),
        },
        "cohort": {
            "subjects": len(subjects),
            "subjects_with_multiple_groups": sum(
                len({str(row["Group"]).strip() for row in visits}) > 1
                for visits in subjects.values()
            ),
            "row_counts_by_group": dict(group_row_counts),
            "unique_subjects_by_group": {
                key: len(value) for key, value in sorted(subjects_by_group.items())
            },
            "row_counts_by_visit": dict(visit_row_counts),
            "descriptions": dict(description_counts),
            "modalities": dict(modality_counts),
            "formats": dict(format_counts),
            "types": dict(type_counts),
        },
        "outputs": {
            "manifest": str(manifest_path),
            "subject_summary": str(subject_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
