#!/usr/bin/env python3
"""
Detect library layout (single-end or paired-end) for SRA accessions.

Uses NCBI's esearch and efetch utilities to query SRA metadata.
Falls back to detecting from downloaded files if metadata query fails.
"""

import sys
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def query_sra_layout(accession: str) -> str:
    """
    Query NCBI SRA database for library layout.

    Returns:
        'PAIRED' or 'SINGLE' or 'UNKNOWN'
    """
    try:
        # Check if EDirect tools are available
        check_esearch = subprocess.run(
            ["which", "esearch"], capture_output=True, timeout=5
        )

        if check_esearch.returncode != 0:
            print(
                f"Warning: esearch not found. Install NCBI EDirect for auto-detection.",
                file=sys.stderr,
            )
            print(f"  conda install -c bioconda entrez-direct", file=sys.stderr)
            return "UNKNOWN"

        # Use esearch to get UID
        esearch_cmd = ["esearch", "-db", "sra", "-query", accession]
        esearch_result = subprocess.run(
            esearch_cmd, capture_output=True, text=True, timeout=30
        )

        if esearch_result.returncode != 0:
            print(f"Warning: esearch failed for {accession}", file=sys.stderr)
            return "UNKNOWN"

        # Use efetch to get XML metadata
        efetch_cmd = ["efetch", "-format", "runinfo", "-mode", "xml"]
        efetch_result = subprocess.run(
            efetch_cmd,
            input=esearch_result.stdout,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if efetch_result.returncode != 0:
            print(f"Warning: efetch failed for {accession}", file=sys.stderr)
            return "UNKNOWN"

        # Parse XML to find LibraryLayout
        root = ET.fromstring(efetch_result.stdout)

        # Try different XML paths (SRA XML schema varies)
        layout = None
        for elem in root.iter("LibraryLayout"):
            layout = elem.text
            break

        # Alternative path
        if not layout:
            for elem in root.iter():
                if "LibraryLayout" in elem.tag:
                    layout = elem.text
                    break

        # Check child elements
        if not layout:
            for elem in root.iter("LibraryLayout"):
                if elem.find("PAIRED") is not None:
                    layout = "PAIRED"
                elif elem.find("SINGLE") is not None:
                    layout = "SINGLE"

        if layout:
            layout = layout.upper()
            if "PAIR" in layout:
                return "PAIRED"
            elif "SINGLE" in layout:
                return "SINGLE"

        return "UNKNOWN"

    except subprocess.TimeoutExpired:
        print(f"Warning: SRA metadata query timed out for {accession}", file=sys.stderr)
        return "UNKNOWN"
    except Exception as e:
        print(
            f"Warning: Error querying SRA metadata for {accession}: {e}",
            file=sys.stderr,
        )
        return "UNKNOWN"


def detect_from_files(data_dir: Path, accession: str) -> str:
    """
    Detect library layout from downloaded FASTQ files.

    Returns:
        'PAIRED' or 'SINGLE' or 'UNKNOWN'
    """
    sample_dir = data_dir / accession

    if not sample_dir.exists():
        return "UNKNOWN"

    # Check for paired-end files
    r1_file = sample_dir / f"{accession}_1.fastq.gz"
    r2_file = sample_dir / f"{accession}_2.fastq.gz"
    se_file = sample_dir / f"{accession}.fastq.gz"

    # Check if files exist and are non-empty
    has_paired = (
        r1_file.exists()
        and r1_file.stat().st_size > 100
        and r2_file.exists()
        and r2_file.stat().st_size > 100
    )
    has_single = se_file.exists() and se_file.stat().st_size > 100

    if has_paired:
        return "PAIRED"
    elif has_single:
        return "SINGLE"
    else:
        return "UNKNOWN"


def main():
    if len(sys.argv) < 2:
        print("Usage: detect_sra_layout.py <accession> [data_dir]", file=sys.stderr)
        sys.exit(1)

    accession = sys.argv[1]
    data_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    # Try querying SRA metadata first (fastest, no download needed)
    layout = query_sra_layout(accession)

    # Fall back to file detection if metadata query failed
    if layout == "UNKNOWN" and data_dir:
        layout = detect_from_files(data_dir, accession)

    # Output result
    print(layout)

    # Exit code: 0 for success, 1 for unknown
    sys.exit(0 if layout != "UNKNOWN" else 1)


if __name__ == "__main__":
    main()
