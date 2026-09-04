"""Cryptographic hashing and artifact integrity verification."""

import hashlib
import os
from typing import Dict, Optional, Tuple, Union
from .constants import KNOWN_HASHES


def compute_sha256(file_path_or_bytes: Union[str, bytes]) -> str:
    """Compute hex SHA-256 hash from a file path or raw bytes."""
    h = hashlib.sha256()
    if isinstance(file_path_or_bytes, bytes):
        h.update(file_path_or_bytes)
        return h.hexdigest()

    file_path = str(file_path_or_bytes)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")

    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_artifact_hash(
    file_path: str,
    expected_hash: Optional[str] = None,
    artifact_key: Optional[str] = None,
) -> Tuple[bool, str, str]:
    """Verify that a given file matches an expected SHA-256 hash.

    Returns:
        Tuple of (is_valid, actual_hash, expected_hash)
    """
    actual_hash = compute_sha256(file_path)
    target_expected = expected_hash
    if target_expected is None and artifact_key is not None:
        target_expected = KNOWN_HASHES.get(artifact_key)

    if target_expected is None:
        return False, actual_hash, "UNKNOWN"

    return (actual_hash.lower() == target_expected.lower()), actual_hash, target_expected
