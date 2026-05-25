#!/usr/bin/env python3
"""
Anonymization scanner for the Provena companion repository.

Recursively scans text files for banned tokens that would break double-blind
anonymization. Exits 0 if clean, nonzero if any violation is found.

Usage:
    python scripts/check_anonymization.py [root_dir]

If root_dir is omitted, scans from the repository root (parent of this script).
"""

import os
import sys

BANNED_TOKENS: list[str] = [
    "AffectLog",
    "AL360",
    "affectlog",
    "al360",
    # Author-identifying names and handles
    "royz.saurabh",
    "github.com/royz",
    "Saurabh",
    "roy-saurabh",
    "github.com/roy-saurabh",
    "0000-0003-3439-7731",
    "orcid.org/0000-0003-3439-7731",
    # Internal product identifiers
    "ai-harbor",
    "ai_harbor",
    # Grant or institutional identifiers
    # (none currently known; add here if applicable)
]

# Files with these extensions are scanned
SCAN_EXTENSIONS: set[str] = {
    ".py", ".md", ".txt", ".json",
    ".yml", ".yaml", ".toml", ".cfg", ".ini",
}

# Directories that should not be scanned
SKIP_DIRS: set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "venv",
    "env",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
}


def is_text_file(path: str) -> bool:
    """Heuristic: try reading first 512 bytes as UTF-8; bail on binary."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(512)
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


# Files that may legitimately contain banned tokens without violating anonymization:
#   - The scanner itself and its tests (they list the tokens by definition)
#   - Public release metadata files: these carry real author info intentionally and
#     must be replaced with their .anonymous counterparts before anonymous submission.
#     The .anonymous files do not contain banned tokens and are not listed here.
SKIP_FILES: set[str] = {
    "check_anonymization.py",
    "test_provena_validation.py",
    "CITATION.cff",
    ".zenodo.json",
    "CITATION.cff.public",
    ".zenodo.public.json",
}


def scan_directory(root: str) -> list[tuple[str, str]]:
    """
    Return list of (filepath, banned_token) for every violation found.
    """
    violations: list[tuple[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            if fname in SKIP_FILES:
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SCAN_EXTENSIONS:
                continue

            fpath = os.path.join(dirpath, fname)
            if not is_text_file(fpath):
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            for token in BANNED_TOKENS:
                if token.lower() in content.lower():
                    violations.append((fpath, token))
                    break  # one violation per file is enough to flag it

    return violations


def main() -> int:
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        # Default: parent directory of this script
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.isdir(root):
        print(f"ERROR: '{root}' is not a directory.", file=sys.stderr)
        return 2

    violations = scan_directory(root)

    if violations:
        print("ANONYMIZATION FAILURE: banned tokens found in repository files.")
        print(f"Scanned from: {root}")
        print()
        for fpath, token in violations:
            rel = os.path.relpath(fpath, root)
            print(f"  {rel}: contains '{token}'")
        print()
        print(
            f"Total violations: {len(violations)}. "
            "Remove or redact all listed occurrences before submission."
        )
        return 1

    print(f"Anonymization check passed. Scanned from: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
