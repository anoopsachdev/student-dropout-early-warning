"""Read the original OULAD CSV release and check that it is the Kuzilek et al. tables."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from oulad.config import EXPECTED_ROWS, RAW_DIR

FILE_NAMES = {
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "vle": "vle.csv",
    "student_info": "studentInfo.csv",
    "student_registration": "studentRegistration.csv",
    "student_assessment": "studentAssessment.csv",
    "student_vle": "studentVle.csv",
}

REQUIRED_COLUMNS = {
    "courses": ["code_module", "code_presentation", "length"],
    "assessments": [
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    ],
    "vle": [
        "id_site",
        "code_module",
        "code_presentation",
        "activity_type",
        "week_from",
        "week_to",
    ],
    "student_info": [
        "code_module",
        "code_presentation",
        "id_student",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "num_of_prev_attempts",
        "studied_credits",
        "disability",
        "final_result",
    ],
    "student_registration": [
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration",
    ],
    "student_assessment": [
        "id_assessment",
        "id_student",
        "date_submitted",
        "is_banked",
        "score",
    ],
    "student_vle": [
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ],
}

STRING_COLUMNS = {
    "code_module",
    "code_presentation",
    "assessment_type",
    "activity_type",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
    "final_result",
}


def ensure_extracted(raw_dir: Path = RAW_DIR) -> None:
    """Unpack anonymisedData.zip into raw_dir if the CSVs are not already there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if (raw_dir / "studentInfo.csv").exists() and (raw_dir / "studentVle.csv").exists():
        return
    candidates = [
        raw_dir / "anonymisedData.zip",
        raw_dir / "oulad_uci.zip",
    ]
    zip_path = next((path for path in candidates if path.exists()), None)
    if zip_path is None:
        raise FileNotFoundError(
            f"Missing {raw_dir / 'studentInfo.csv'}. Run scripts/download_oulad.py first."
        )
    staging = raw_dir / "_zip"
    staging.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (staging / member.filename).resolve()
            if not str(target).startswith(str(staging.resolve())):
                raise RuntimeError(f"Unsafe zip path: {member.filename}")
        archive.extractall(staging)
    for csv_path in staging.rglob("*.csv"):
        destination = raw_dir / csv_path.name
        if not destination.exists():
            csv_path.replace(destination)
    for child in sorted(staging.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            child.rmdir()


def _clean_strings(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column not in STRING_COLUMNS:
            continue
        out[column] = out[column].astype("string").str.strip()
        out.loc[out[column].isin({"", "<NA>", "nan", "None"}), column] = pd.NA
    return out


def normalize_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Cast keys and rename the course-length column. Drops the Kaggle-only vle.title field."""
    missing = [name for name in FILE_NAMES if name not in tables]
    if missing:
        raise KeyError(f"Missing tables: {missing}")
    cleaned = {}
    for name, required in REQUIRED_COLUMNS.items():
        frame = tables[name].copy()
        frame.columns = [str(column).strip() for column in frame.columns]
        if name == "courses" and "module_presentation_length" in frame.columns:
            frame = frame.rename(columns={"module_presentation_length": "length"})
        if name == "vle" and "title" in frame.columns:
            frame = frame.drop(columns=["title"])
            print("Dropped vle.title. That column is not in the OU release.", flush=True)
        absent = [column for column in required if column not in frame.columns]
        if absent:
            raise ValueError(f"{name} is missing columns {absent}")
        frame = frame[required]
        frame = _clean_strings(frame)
        cleaned[name] = frame
    courses = cleaned["courses"].rename(columns={"length": "module_length"})
    courses["module_length"] = pd.to_numeric(courses["module_length"], errors="coerce")
    cleaned["courses"] = courses
    assessments = cleaned["assessments"]
    assessments["id_assessment"] = pd.to_numeric(assessments["id_assessment"], errors="raise").astype("int64")
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["weight"] = pd.to_numeric(assessments["weight"], errors="coerce")
    cleaned["assessments"] = assessments.rename(columns={"date": "date_due"})
    for key in ("id_student",):
        for name in ("student_info", "student_registration", "student_assessment", "student_vle"):
            if key in cleaned[name].columns:
                cleaned[name][key] = pd.to_numeric(cleaned[name][key], errors="raise").astype("int64")
    vle = cleaned["vle"]
    vle["id_site"] = pd.to_numeric(vle["id_site"], errors="raise").astype("int64")
    cleaned["vle"] = vle
    clicks = cleaned["student_vle"]
    clicks["id_site"] = pd.to_numeric(clicks["id_site"], errors="raise").astype("int64")
    clicks["date"] = pd.to_numeric(clicks["date"], errors="raise").astype("int32")
    clicks["sum_click"] = pd.to_numeric(clicks["sum_click"], errors="raise").astype("int32")
    cleaned["student_vle"] = clicks
    info = cleaned["student_info"]
    info["num_of_prev_attempts"] = pd.to_numeric(info["num_of_prev_attempts"], errors="coerce")
    info["studied_credits"] = pd.to_numeric(info["studied_credits"], errors="coerce")
    cleaned["student_info"] = info
    registration = cleaned["student_registration"]
    registration["date_registration"] = pd.to_numeric(registration["date_registration"], errors="coerce")
    registration["date_unregistration"] = pd.to_numeric(
        registration["date_unregistration"], errors="coerce"
    )
    cleaned["student_registration"] = registration
    submitted = cleaned["student_assessment"]
    submitted["id_assessment"] = pd.to_numeric(submitted["id_assessment"], errors="raise").astype("int64")
    submitted["date_submitted"] = pd.to_numeric(submitted["date_submitted"], errors="coerce")
    submitted["is_banked"] = pd.to_numeric(submitted["is_banked"], errors="coerce").fillna(0).astype("int8")
    submitted["score"] = pd.to_numeric(submitted["score"], errors="coerce")
    cleaned["student_assessment"] = submitted
    return cleaned


def validate_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, expected in EXPECTED_ROWS.items():
        actual = len(tables[name])
        if actual != expected:
            raise ValueError(
                f"{name} has {actual} rows; the OU release has {expected}. "
                "Refusing to continue with a different extract."
            )
    results = set(tables["student_info"]["final_result"].dropna().unique())
    expected_results = {"Distinction", "Pass", "Fail", "Withdrawn"}
    if results != expected_results:
        raise ValueError(f"Unexpected final_result values: {sorted(results)}")
    print(
        "Loaded OULAD "
        f"({len(tables['student_info'])} registrations, "
        f"{len(tables['student_vle'])} VLE rows, "
        f"{len(tables['courses'])} presentations).",
        flush=True,
    )


def load_tables(raw_dir: Path = RAW_DIR, validate: bool = True) -> dict[str, pd.DataFrame]:
    ensure_extracted(raw_dir)
    tables = {}
    for name, filename in FILE_NAMES.items():
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Reading {filename}", flush=True)
        tables[name] = pd.read_csv(path, na_values=["?"], keep_default_na=True)
    tables = normalize_tables(tables)
    if validate:
        validate_tables(tables)
    return tables
