"""Data cleaning, normalization, and auditing pipeline."""

from typing import Any, Dict, List, Optional, Set, Tuple
import unicodedata
from collections import Counter, defaultdict
from .constants import ALLOWED_LABELS_SET, LABEL_ALIAS_MAP


def normalize_text(text: Any) -> str:
    """Normalize unicode with NFKC and collapse contiguous whitespace."""
    if text is None:
        return ""
    norm = unicodedata.normalize("NFKC", str(text).strip())
    return " ".join(norm.split())


def normalize_label(label: Any) -> Optional[str]:
    """Canonicalize label string to SUPPORTS, REFUTES, or NOT_ENOUGH_INFO."""
    if label is None:
        return None
    raw_str = normalize_text(label)
    if raw_str in LABEL_ALIAS_MAP:
        return LABEL_ALIAS_MAP[raw_str]
    # Check upper case
    upper_str = raw_str.upper()
    if upper_str in LABEL_ALIAS_MAP:
        return LABEL_ALIAS_MAP[upper_str]
    return None


def normalize_evidence_list(evidence: Any) -> List[str]:
    """Convert evidence into a deduplicated list of non-empty cleaned passages."""
    passages: List[str] = []
    if isinstance(evidence, str):
        cleaned = normalize_text(evidence)
        if cleaned:
            passages.append(cleaned)
    elif isinstance(evidence, (list, tuple)):
        seen: Set[str] = set()
        for item in evidence:
            cleaned = normalize_text(item)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                passages.append(cleaned)
    return passages


def clean_record(record: Dict[str, Any], is_training: bool = True) -> Optional[Dict[str, Any]]:
    """Clean a single record dictionary. Returns None if unrecoverable."""
    rec_id = str(record.get("id", "")).strip()
    claim = normalize_text(record.get("claim", ""))
    evidence = normalize_evidence_list(record.get("evidence", []))

    if not claim:
        return None

    cleaned_rec: Dict[str, Any] = {
        "id": rec_id,
        "claim": claim,
        "evidence": evidence,
    }

    if is_training or "label" in record:
        raw_label = record.get("label")
        canonical_label = normalize_label(raw_label)
        if canonical_label is None or canonical_label not in ALLOWED_LABELS_SET:
            return None
        cleaned_rec["label"] = canonical_label

    return cleaned_rec


def clean_dataset(
    records: List[Dict[str, Any]],
    is_training: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute complete 10-step data cleaning and auditing pipeline.

    Returns:
        Tuple of (cleaned_records, audit_report)
    """
    raw_count = len(records)
    stats: Dict[str, Any] = {
        "raw_rows": raw_count,
        "dropped_missing_or_invalid_label": 0,
        "dropped_exact_duplicates": 0,
        "dropped_conflicting_duplicates": 0,
        "cleaned_rows": 0,
    }

    # Step 1-7: Single-record normalization & filtering
    first_pass: List[Dict[str, Any]] = []
    for r in records:
        cleaned = clean_record(r, is_training=is_training)
        if cleaned is None:
            stats["dropped_missing_or_invalid_label"] += 1
            continue
        first_pass.append(cleaned)

    # Step 8-9: Deduplication & Conflict Detection
    # Group by normalized (claim, evidence_tuple)
    grouped: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for r in first_pass:
        key = (r["claim"], tuple(r["evidence"]))
        grouped[key].append(r)

    deduped_records: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        if len(group) == 1:
            deduped_records.append(group[0])
            continue

        if is_training:
            # Check if conflicting labels exist in group
            distinct_labels = {g.get("label") for g in group}
            if len(distinct_labels) > 1:
                # Conflicting duplicates -> drop group entirely
                stats["dropped_conflicting_duplicates"] += len(group)
                continue
            else:
                # Exact duplicates with same label -> keep only first
                stats["dropped_exact_duplicates"] += (len(group) - 1)
                deduped_records.append(group[0])
        else:
            # For inference / test sets, retain entries with distinct IDs
            seen_ids = set()
            for g in group:
                if g["id"] not in seen_ids:
                    seen_ids.add(g["id"])
                    deduped_records.append(g)

    stats["cleaned_rows"] = len(deduped_records)
    if is_training:
        stats["label_distribution"] = dict(Counter(r["label"] for r in deduped_records))

    return deduped_records, stats
