#!/usr/bin/env python3
"""Delete loose files sitting directly in a directory (not inside any subdirectory).

Dry run by default — nothing is deleted unless you pass --apply.

Examples:
    python clear_unorganized.py data                 # preview .csv files that would go
    python clear_unorganized.py data --apply         # actually delete them
    python clear_unorganized.py data --all --apply   # delete every loose file, not just .csv
"""

import argparse
import sys
from pathlib import Path


def find_loose_files(root: Path, extensions: list[str] | None) -> list[Path]:
    """Return files directly inside `root`, ignoring anything in subdirectories."""
    files = []
    for entry in sorted(root.iterdir()):
        if not entry.is_file() or entry.is_symlink():
            continue
        if extensions and entry.suffix.lower() not in extensions:
            continue
        files.append(entry)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", nargs="?", default="data",
                        help="directory to clean (default: data)")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without this it's a dry run")
    parser.add_argument("--all", action="store_true",
                        help="delete every loose file, not just .csv")
    parser.add_argument("--ext", default=".csv",
                        help="comma-separated extensions to target (default: .csv)")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="skip the confirmation prompt")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    extensions = None
    if not args.all:
        extensions = [e if e.startswith(".") else "." + e
                      for e in (x.strip().lower() for x in args.ext.split(",")) if e]

    targets = find_loose_files(root, extensions)

    if not targets:
        print(f"Nothing to clean in {root}")
        return 0

    total = sum(f.stat().st_size for f in targets)
    print(f"{len(targets)} loose file(s) in {root} ({total / 1024:.1f} KiB):")
    for f in targets:
        print(f"  {f.name}")

    if not args.apply:
        print("\nDry run — nothing deleted. Re-run with --apply to delete.")
        return 0

    if not args.yes:
        answer = input(f"\nPermanently delete these {len(targets)} file(s)? [y/N] ")
        if answer.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    deleted = 0
    for f in targets:
        try:
            f.unlink()
            deleted += 1
        except OSError as exc:
            print(f"Could not delete {f.name}: {exc}", file=sys.stderr)

    print(f"Deleted {deleted} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
