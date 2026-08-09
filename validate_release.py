from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FORBIDDEN_DIRECTORIES = {
    "adni_work",
    "data",
    "outputs",
    "results",
    "checkpoints",
    "private",
}
FORBIDDEN_SUFFIXES = {
    ".nii",
    ".dcm",
    ".mat",
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".weights",
    ".log",
}
TEXT_SUFFIXES = {".m", ".py", ".ps1", ".md", ".cff", ".txt"}
REQUIRED_FILES = {
    "README.md",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    "CITATION.cff",
    "REPOSITORY_METADATA.md",
    "DATA_ACCESS.md",
    "run_adni_resnet3d18_pipeline.ps1",
    "run_adni_resnet3d18_cv_v4_phase3_recovery.ps1",
    "run_adni_resnet3d18_cv_v4_phase4.ps1",
    "run_adni_resnet3d18_cv_v4_paired_rtx5070.ps1",
    "ADNI_NMA_CV_V4_LOCKED_INTERNAL_VALIDATION_PROTOCOL.md",
    "ADNI_NMA_CV_V4_ENGINEERING_SMOKE_PROTOCOL.md",
    "ADNI_NMA_CV_V4_PHASE3_RECOVERY_AMENDMENT.md",
    "ADNI_NMA_CV_V4_PHASE4_ATOMIC_OUTER_TEST_PROTOCOL.md",
    "ADNI_NMA_CV_V4_PAIRED_RTX5070_BENCHMARK_PROTOCOL.md",
    "matlab/nma_allocate_single_resource.m",
    "matlab/nma_demo_allocator.m",
    "matlab/nma_resnet3d18_allocate_multiresource.m",
    "matlab/adni_run_resnet3d18_cv_v4_validation_cell.m",
    "matlab/analyze_adni_resnet3d18_cv_v4_outer_test.m",
    "matlab/aggregate_adni_resnet3d18_cv_v4_paired_rtx5070.m",
}


def repository_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    )


def main() -> int:
    failures: list[str] = []
    files = repository_files()
    relative_files = {path.relative_to(ROOT).as_posix() for path in files}

    missing = sorted(REQUIRED_FILES - relative_files)
    if missing:
        failures.append(f"Missing required files: {missing}")

    forbidden_directories = sorted(
        {
            part
            for path in files
            for part in path.relative_to(ROOT).parts[:-1]
            if part.lower() in FORBIDDEN_DIRECTORIES
        }
    )
    if forbidden_directories:
        failures.append(
            f"Forbidden generated/data directories: {forbidden_directories}"
        )

    forbidden_files = []
    for path in files:
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if lower_name.endswith(".nii.gz") or suffix in FORBIDDEN_SUFFIXES:
            forbidden_files.append(path.relative_to(ROOT).as_posix())
    if forbidden_files:
        failures.append(f"Forbidden data/model artifacts: {forbidden_files}")

    absolute_path_patterns = [
        re.compile(r"[DF]:\\(?:ADNI|AIs)\\", re.IGNORECASE),
        re.compile(r"C:\\Users\\", re.IGNORECASE),
    ]
    participant_pattern = re.compile(
        r"\b" + r"\d{3}" + "_S_" + r"\d{4}\b", re.IGNORECASE
    )
    absolute_path_hits: list[str] = []
    participant_hits: list[str] = []
    python_failures: list[str] = []

    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            ".gitignore",
            ".gitattributes",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT).as_posix()
        if any(pattern.search(text) for pattern in absolute_path_patterns):
            absolute_path_hits.append(relative)
        if participant_pattern.search(text):
            participant_hits.append(relative)
        if path.suffix.lower() == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as error:
                python_failures.append(f"{relative}: {error}")

    if absolute_path_hits:
        failures.append(f"Personal/local absolute paths: {absolute_path_hits}")
    if participant_hits:
        failures.append(f"Potential ADNI participant identifiers: {participant_hits}")
    if python_failures:
        failures.append(f"Python syntax failures: {python_failures}")

    metadata = (ROOT / "REPOSITORY_METADATA.md").read_text(encoding="utf-8")
    description_match = re.search(
        r"## GitHub description[^\n]*\n\n([^\n]+)", metadata
    )
    description_length = len(description_match.group(1)) if description_match else 0
    if not description_match or description_length > 350:
        failures.append(
            f"GitHub description missing or exceeds 350 characters: {description_length}"
        )

    matlab_files = sorted((ROOT / "matlab").glob("*.m"))
    if len(matlab_files) != 49:
        failures.append(f"Expected 49 MATLAB files, found {len(matlab_files)}")

    report = {
        "status": "PASS" if not failures else "FAIL",
        "repository": "https://github.com/msaitd/neural-murray-wbe-scaling",
        "files_checked": len(files),
        "matlab_files": len(matlab_files),
        "github_description_characters": description_length,
        "forbidden_data_or_model_files": len(forbidden_files),
        "local_absolute_path_hits": len(absolute_path_hits),
        "participant_identifier_hits": len(participant_hits),
        "python_syntax_failures": len(python_failures),
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
