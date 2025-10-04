import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

# --- Function to load saved sparse k-mer results ---
# (Copied and slightly adapted from the main k-mer counting script)


logger = logging.getLogger(__name__)


def load_kmer_results(base_path, verbose=True):
    """
    Loads k-mer results that were saved in the sparse format
    (matrix.npz, rows.txt, cols.txt).

    Args:
        base_path (str): The base path used when saving (e.g., 'output/my_kmers_binary').
                         The function will look for '{base_path}_sparse.npz', etc.
                         It automatically checks for the '_binary' suffix if the
                         non-suffixed files aren't found.
        verbose (bool): If True, print loading messages.

    Returns:
        tuple: (scipy.sparse.csr_matrix, list of row names/IDs, list of col names/k-mers)
               Returns (None, None, None) if files are not found or an error occurs.
    """
    # Determine file paths, checking for standard and _binary versions
    sparse_matrix_file = f"{base_path}_sparse.npz"
    rows_file = f"{base_path}_rows.txt"
    cols_file = f"{base_path}_cols.txt"
    is_binary = False  # Flag to track which set of files is used

    if not (
        os.path.exists(sparse_matrix_file)
        and os.path.exists(rows_file)
        and os.path.exists(cols_file)
    ):
        # Try with _binary suffix if primary files not found
        binary_sparse_matrix_file = f"{base_path}_binary_sparse.npz"
        binary_rows_file = f"{base_path}_binary_rows.txt"
        binary_cols_file = f"{base_path}_binary_cols.txt"
        if (
            os.path.exists(binary_sparse_matrix_file)
            and os.path.exists(binary_rows_file)
            and os.path.exists(binary_cols_file)
        ):
            sparse_matrix_file = binary_sparse_matrix_file
            rows_file = binary_rows_file
            cols_file = binary_cols_file
            is_binary = True
            if verbose:
                logger.info(f"Found binary k-mer results at: {base_path}_binary*")
        else:
            if verbose:
                print(
                    f"Error: One or more required sparse result files not found for base path '{base_path}'.\n"
                    f"Checked for:\n"
                    f" - {base_path}_sparse.npz / _rows.txt / _cols.txt\n"
                    f" - {base_path}_binary_sparse.npz / _binary_rows.txt / _binary_cols.txt",
                    file=sys.stderr,
                )
            return None, None, None
    else:
        if verbose:
            logger.info(f"Found standard k-mer results at: {base_path}*")

    if verbose:
        logger.info(f"Loading sparse matrix from: {sparse_matrix_file}")
        logger.info(f"Loading row names from: {rows_file}")
        logger.info(f"Loading column names from: {cols_file}")

    try:
        # Load the sparse matrix (usually saved in CSR format)
        sparse_matrix = load_npz(sparse_matrix_file)
        # Load row names (sequence IDs)
        with open(rows_file, "r") as f:
            row_names = [line.strip() for line in f if line.strip()]  # Read and strip whitespace
        # Load column names (k-mers)
        with open(cols_file, "r") as f:
            col_names = [line.strip() for line in f if line.strip()]  # Read and strip whitespace

        # --- Validation ---
        if sparse_matrix.shape[0] != len(row_names):
            logger.info(
                f"Error: Matrix row count ({sparse_matrix.shape[0]}) does not match number of row names ({len(row_names)}) in {rows_file}.",
                file=sys.stderr,
            )
            return None, None, None
        if sparse_matrix.shape[1] != len(col_names):
            logger.info(
                f"Error: Matrix column count ({sparse_matrix.shape[1]}) does not match number of column names ({len(col_names)}) in {cols_file}.",
                file=sys.stderr,
            )
            return None, None, None

        if verbose:
            logger.info(f"Loaded sparse matrix with shape {sparse_matrix.shape}.")
            logger.info(f"Loaded {len(row_names)} row names and {len(col_names)} column names.")

        return sparse_matrix, row_names, col_names
    except FileNotFoundError as e:
        if verbose:
            logger.info(
                f"Error: A required file was not found during loading: {e}", file=sys.stderr
            )
        return None, None, None
    except Exception as e:
        if verbose:
            logger.info(f"Error loading k-mer results: {e}", file=sys.stderr)
        return None, None, None


# --- Command-line Argument Parsing ---
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert saved sparse k-mer results (.npz, .txt files) to a dense Parquet file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_base",
        required=True,
        help=(
            "Base path of the saved sparse k-mer results. "
            "The script expects files like '{input_base}_sparse.npz', "
            "'{input_base}_rows.txt', '{input_base}_cols.txt' (or their '_binary' equivalents)."
        ),
    )
    parser.add_argument(
        "-o", "--output_parquet", required=True, help="Path for the output dense Parquet file."
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress progress messages.")
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["snappy", "gzip", "brotli", "zstd", "lz4", None],  # Common Parquet compressions
        help="Compression algorithm to use for the output Parquet file. Use None for no compression.",
    )
    return parser.parse_args()


# --- Main Execution ---
if __name__ == "__main__":
    # Check for pyarrow dependency first
    try:
        import pyarrow

        # if hasattr(pyarrow, 'version'): # Basic check
        #      pass # pyarrow seems available
        # else: # Handle potential edge cases where import succeeds but module is incomplete
        #      print("Error: Imported 'pyarrow', but it seems incomplete or unusable.", file=sys.stderr)
        #      sys.exit(1)
    except ImportError:
        logger.info(
            "Error: This script requires the 'pyarrow' library to write Parquet files.",
            file=sys.stderr,
        )
        logger.info("Please install it, for example using: pip install pyarrow", file=sys.stderr)
        sys.exit(1)

    args = parse_arguments()
    verbose = not args.quiet

    start_time = time.time()

    if verbose:
        logger.info("--- Sparse K-mer to Dense Parquet Converter ---")
        logger.info(f"Input Base Path: {args.input_base}")
        logger.info(f"Output Parquet File: {args.output_parquet}")
        logger.info(f"Compression: {args.compression}")
        logger.info("---------------------------------------------")

    # 1. Load the sparse data
    sparse_matrix, row_names, col_names = load_kmer_results(args.input_base, verbose=verbose)

    if sparse_matrix is None:
        logger.info("Failed to load sparse data. Exiting.", file=sys.stderr)
        sys.exit(1)

    # 2. Convert to Dense DataFrame
    if verbose:
        logger.info("\nConverting sparse matrix to dense Pandas DataFrame...")
        logger.info(
            f"(Matrix dimensions: {sparse_matrix.shape[0]} rows x {sparse_matrix.shape[1]} columns)"
        )
        num_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
        if num_elements > 10**8:  # Warn if potentially large
            logger.info(
                "Warning: Matrix dimensions are large. Conversion to dense format might require significant memory.",
                file=sys.stderr,
            )

    try:
        # Convert sparse matrix to dense NumPy array first
        dense_array = sparse_matrix.toarray()
        # Create DataFrame
        df_dense = pd.DataFrame(dense_array, index=row_names, columns=col_names)
        del dense_array  # Free memory from the intermediate dense array if possible
        if verbose:
            dense_mem_mb = df_dense.memory_usage(deep=True).sum() / (1024**2)
            logger.info(
                f"Dense DataFrame created successfully. Estimated memory usage: {dense_mem_mb:.2f} MB"
            )

    except MemoryError:
        logger.info(
            "\nERROR: A MemoryError occurred while converting the sparse matrix to a dense DataFrame.",
            file=sys.stderr,
        )
        logger.info(
            "The dataset is too large to fit into memory in a dense format.", file=sys.stderr
        )
        logger.info(
            "Consider processing the sparse matrix directly if possible, or using a machine with more RAM.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as e:
        logger.info(f"\nAn unexpected error occurred during dense conversion: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Save to Parquet
    if verbose:
        logger.info(f"\nSaving dense DataFrame to Parquet file: {args.output_parquet}")
        logger.info(f"Using compression: {args.compression if args.compression else 'None'}")

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output_parquet)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            if verbose:
                logger.info(f"Created output directory: {output_dir}")

        # Save using pyarrow engine
        df_dense.to_parquet(
            args.output_parquet,
            engine="pyarrow",
            compression=args.compression,  # Pass user's choice or None
            index=True,  # Include the sequence IDs (index) in the Parquet file
        )
        if verbose:
            logger.info("Parquet file saved successfully.")

    except Exception as e:
        logger.info(f"\nError saving DataFrame to Parquet file: {e}", file=sys.stderr)
        sys.exit(1)

    end_time = time.time()
    if verbose:
        logger.info(f"\nConversion finished in {end_time - start_time:.2f} seconds.")
