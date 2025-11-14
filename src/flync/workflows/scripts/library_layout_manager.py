#!/usr/bin/env python3
"""
Library Layout Manager for FLYNC Pipeline

Manages per-sample library layout (paired-end vs single-end) detection and storage.
Supports three modes:
1. Global setting (fastq_paired: true/false in config)
2. Mapping file (sample_id -> layout mapping)
3. Auto-detection (from SRA metadata or file patterns)
"""

import sys
import json
from pathlib import Path
from typing import Dict, Optional


class LibraryLayoutManager:
    """Manages library layout information for samples."""

    def __init__(self, layout_file: Path):
        """
        Initialize manager with path to layout storage file.

        Args:
            layout_file: Path to JSON file storing sample layouts
        """
        self.layout_file = Path(layout_file)
        self._layouts: Dict[str, bool] = {}
        self._load()

    def _load(self):
        """Load existing layout data from file."""
        if self.layout_file.exists():
            try:
                with open(self.layout_file, "r") as f:
                    self._layouts = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load layout file: {e}", file=sys.stderr)
                self._layouts = {}

    def _save(self):
        """Save layout data to file."""
        self.layout_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.layout_file, "w") as f:
            json.dump(self._layouts, f, indent=2)

    def set_layout(self, sample_id: str, is_paired: bool):
        """
        Set layout for a sample.

        Args:
            sample_id: Sample identifier
            is_paired: True if paired-end, False if single-end
        """
        self._layouts[sample_id] = is_paired
        self._save()

    def get_layout(self, sample_id: str) -> Optional[bool]:
        """
        Get layout for a sample.

        Args:
            sample_id: Sample identifier

        Returns:
            True if paired-end, False if single-end, None if unknown
        """
        return self._layouts.get(sample_id)

    def is_paired(self, sample_id: str) -> bool:
        """
        Check if sample is paired-end (defaults to False if unknown).

        Args:
            sample_id: Sample identifier

        Returns:
            True if paired-end, False otherwise
        """
        return self._layouts.get(sample_id, False)

    def has_layout(self, sample_id: str) -> bool:
        """
        Check if layout is known for sample.

        Args:
            sample_id: Sample identifier

        Returns:
            True if layout is known
        """
        return sample_id in self._layouts

    def bulk_set_layouts(self, layouts: Dict[str, bool]):
        """
        Set layouts for multiple samples at once.

        Args:
            layouts: Dictionary mapping sample_id -> is_paired
        """
        self._layouts.update(layouts)
        self._save()

    def get_all_layouts(self) -> Dict[str, bool]:
        """Get all stored layouts."""
        return self._layouts.copy()


def load_layout_mapping(mapping_file: Path) -> Dict[str, bool]:
    """
    Load library layout mapping from file.

    Supports CSV format:
        sample_id,paired
        SRR123456,true
        SRR123457,false

    Args:
        mapping_file: Path to mapping file

    Returns:
        Dictionary mapping sample_id -> is_paired
    """
    import csv

    layouts = {}

    with open(mapping_file, "r") as f:
        # Try CSV format first
        if mapping_file.suffix in [".csv", ".CSV"]:
            reader = csv.DictReader(f)
            for row in reader:
                if "sample_id" not in row or "paired" not in row:
                    raise ValueError(
                        f"Layout mapping file must have 'sample_id' and 'paired' columns. "
                        f"Found columns: {list(row.keys())}"
                    )

                sample_id = row["sample_id"].strip()
                paired_str = row["paired"].strip().lower()

                # Parse boolean values
                if paired_str in ["true", "1", "yes", "paired"]:
                    is_paired = True
                elif paired_str in ["false", "0", "no", "single"]:
                    is_paired = False
                else:
                    raise ValueError(
                        f"Invalid paired value for {sample_id}: '{paired_str}'. "
                        f"Use: true/false, 1/0, yes/no, or paired/single"
                    )

                layouts[sample_id] = is_paired
        else:
            # Simple format: sample_id<TAB>paired (true/false)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split("\t") if "\t" in line else line.split()
                if len(parts) != 2:
                    raise ValueError(f"Invalid line in layout mapping: '{line}'")

                sample_id = parts[0].strip()
                paired_str = parts[1].strip().lower()

                if paired_str in ["true", "1", "yes", "paired"]:
                    is_paired = True
                elif paired_str in ["false", "0", "no", "single"]:
                    is_paired = False
                else:
                    raise ValueError(
                        f"Invalid paired value for {sample_id}: '{paired_str}'"
                    )

                layouts[sample_id] = is_paired

    return layouts


def main():
    """CLI for library layout manager."""
    if len(sys.argv) < 3:
        print(
            "Usage: library_layout_manager.py <layout_file> <command> [args...]",
            file=sys.stderr,
        )
        print("\nCommands:", file=sys.stderr)
        print("  get <sample_id>              - Get layout for sample", file=sys.stderr)
        print("  set <sample_id> <true|false> - Set layout for sample", file=sys.stderr)
        print(
            "  load <mapping_file>          - Load layouts from mapping file",
            file=sys.stderr,
        )
        print("  list                         - List all layouts", file=sys.stderr)
        sys.exit(1)

    layout_file = Path(sys.argv[1])
    command = sys.argv[2]

    manager = LibraryLayoutManager(layout_file)

    if command == "get":
        if len(sys.argv) < 4:
            print("Error: sample_id required", file=sys.stderr)
            sys.exit(1)
        sample_id = sys.argv[3]
        layout = manager.get_layout(sample_id)
        if layout is None:
            print("UNKNOWN")
            sys.exit(1)
        else:
            print("PAIRED" if layout else "SINGLE")

    elif command == "set":
        if len(sys.argv) < 5:
            print("Error: sample_id and paired status required", file=sys.stderr)
            sys.exit(1)
        sample_id = sys.argv[3]
        is_paired = sys.argv[4].lower() in ["true", "1", "yes", "paired"]
        manager.set_layout(sample_id, is_paired)
        print(f"Set {sample_id}: {'PAIRED' if is_paired else 'SINGLE'}")

    elif command == "load":
        if len(sys.argv) < 4:
            print("Error: mapping_file required", file=sys.stderr)
            sys.exit(1)
        mapping_file = Path(sys.argv[3])
        layouts = load_layout_mapping(mapping_file)
        manager.bulk_set_layouts(layouts)
        print(f"Loaded {len(layouts)} layouts from {mapping_file}")

    elif command == "list":
        layouts = manager.get_all_layouts()
        for sample_id, is_paired in sorted(layouts.items()):
            print(f"{sample_id}\t{'PAIRED' if is_paired else 'SINGLE'}")

    else:
        print(f"Error: Unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
