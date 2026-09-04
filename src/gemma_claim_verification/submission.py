"""Competition submission file generation and validation."""

import csv
import os
from typing import Any, Dict, List, Optional
from .constants import ALLOWED_LABELS_SET


def validate_submission_file(
    csv_path: str,
    expected_ids: Optional[List[str]] = None,
    expected_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Strictly validate a generated submission CSV file.

    Asserts:
        - File exists and is non-empty
        - Exact header: ['id', 'label']
        - Expected row count
        - All IDs unique and in matching order if expected_ids provided
        - All labels strictly in ALLOWED_LABELS_SET
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Submission file does not exist: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"Invalid submission header: {reader.fieldnames}. Expected exact ['id', 'label'].")
        for row in reader:
            rows.append(row)

    row_count = len(rows)
    if expected_count is not None and row_count != expected_count:
        raise ValueError(f"Row count mismatch: expected {expected_count}, found {row_count}.")

    seen_ids = set()
    for idx, r in enumerate(rows):
        rec_id = r.get("id", "").strip()
        label = r.get("label", "").strip()

        if not rec_id:
            raise ValueError(f"Empty ID found at row {idx + 1}")
        if rec_id in seen_ids:
            raise ValueError(f"Duplicate ID '{rec_id}' found at row {idx + 1}")
        seen_ids.add(rec_id)

        if label not in ALLOWED_LABELS_SET:
            raise ValueError(f"Invalid label '{label}' at row {idx + 1}. Must be one of {ALLOWED_LABELS_SET}")

        if expected_ids is not None and idx < len(expected_ids):
            if rec_id != expected_ids[idx]:
                raise ValueError(f"ID order mismatch at row {idx + 1}: expected '{expected_ids[idx]}', got '{rec_id}'")

    return {
        "status": "VALID",
        "row_count": row_count,
        "unique_ids": len(seen_ids),
        "label_distribution": dict(Counter(r["label"] for r in rows)),
    }


def create_submission_file(
    predictions: List[Dict[str, Any]],
    output_csv_path: str = "outputs/submission.csv",
    expected_ids: Optional[List[str]] = None,
) -> str:
    """Generate and validate a submission CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"])
        for p in predictions:
            rec_id = str(p["id"]).strip()
            label = str(p["prediction"]).strip()
            writer.writerow([rec_id, label])

    # Re-read and validate immediately
    validate_submission_file(
        output_csv_path,
        expected_ids=expected_ids,
        expected_count=len(predictions),
    )
    print(f"Submission successfully created and validated at: {output_csv_path}")
    return output_csv_path


from collections import Counter
