#!/usr/bin/env python3
"""CLI script to verify SHA-256 integrity of project artifacts and datasets."""

import argparse
import os
import sys

from gemma_claim_verification.constants import KNOWN_HASHES
from gemma_claim_verification.hashing import compute_sha256, verify_artifact_hash


def main():
    parser = argparse.ArgumentParser(description="Verify SHA-256 hash of an artifact.")
    parser.add_argument("file_path", help="Path to the file to verify.")
    parser.add_argument("--key", "-k", choices=list(KNOWN_HASHES.keys()), default=None, help="Known artifact key.")
    parser.add_argument("--expected", "-e", default=None, help="Explicit expected SHA-256 string.")
    args = parser.parse_args()

    actual_hash = compute_sha256(args.file_path)
    print(f"File:        {args.file_path}")
    print(f"SHA-256:     {actual_hash}")

    target = args.expected or (KNOWN_HASHES.get(args.key) if args.key else None)
    if target:
        print(f"Expected:    {target}")
        if actual_hash.lower() == target.lower():
            print("Status:      [PASS] Exact hash match.")
            sys.exit(0)
        else:
            print("Status:      [FAIL] Hash mismatch!")
            sys.exit(1)
    else:
        # Check against all known hashes
        matched_keys = [k for k, v in KNOWN_HASHES.items() if v.lower() == actual_hash.lower()]
        if matched_keys:
            print(f"Status:      [MATCH] Matches known artifact key: {matched_keys[0]}")
        else:
            print("Status:      [INFO] Unrecognized SHA-256 (not in known competition table).")


if __name__ == "__main__":
    main()
