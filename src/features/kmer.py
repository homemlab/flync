import pandas as pd
from Bio import SeqIO
from collections import Counter
import itertools
import argparse
import sys
import os
import time
import logging
import tempfile
import uuid
from multiprocessing import Pool, cpu_count

# Need vstack for combining sparse matrices
from scipy.sparse import dok_matrix, save_npz, load_npz, vstack
import numpy as np
import glob
import hashlib
import tempfile  # For temporary directory

# Import progress utilities
try:
    from src.utils.progress import (
        get_progress_manager,
        update_cli_args,
        resolve_progress_settings,
        create_progress_bar,
    )
except ImportError:
    # Try relative import when running as script
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from utils.progress import (
        get_progress_manager,
        update_cli_args,
        resolve_progress_settings,
        create_progress_bar,
    )

__all__ = ["calculate_kmer_profiles", "load_kmer_results"]  # Exportable functions


# --- Logging Setup ---
def setup_logging(log_level="INFO"):
    """
    Configures the root logger for the application.

    Args:
        log_level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
                         Defaults to 'INFO'.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        # Default to INFO if an invalid log level string is provided
        print(
            f"Warning: Invalid log level '{log_level}'. Defaulting to INFO.",
            file=sys.stderr,
        )
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        stream=sys.stdout,  # Direct logs to stdout
    )
    return logging.getLogger(__name__)


# --- Helper Function to Convert DNA to Binary ---
def _dna_to_binary(dna_sequence):
    """
    Converts a DNA/RNA sequence string (ATGC/AUGC) to its binary representation.

    A, T, U -> 0
    G, C -> 1

    Other characters (e.g., 'N', ambiguous codes) are ignored and will not
    be part of the resulting binary string.

    Args:
        dna_sequence (str): A DNA or RNA sequence string.

    Returns:
        str: A binary string representation ('0's and '1's).
    """
    dna_sequence = dna_sequence.upper()
    binary_seq = []
    for base in dna_sequence:
        if base in ("A", "T", "U"):  # Add 'U' for RNA support
            binary_seq.append("0")
        elif base in ("G", "C"):
            binary_seq.append("1")
    return "".join(binary_seq)


# --- Helper Functions for K-mer Operations ---
def _generate_all_kmers(k_min, k_max, alphabet="01", verbose=True):
    """
    Generates all possible unique k-mers for a given range of k values and alphabet.
    For binary k-mers, alphabet should be '01'.

    Args:
        k_min (int): Minimum k-mer length.
        k_max (int): Maximum k-mer length.
        alphabet (str): The characters to use for generating k-mers.
        verbose (bool): If True, log progress messages.

    Returns:
        list: A sorted list of unique k-mers.
    """
    all_kmers = set()
    t_start = time.time()
    if verbose:
        logging.info(
            f"Generating all possible k-mers for k={k_min}-{k_max} using alphabet '{alphabet}'..."
        )
    for k in range(k_min, k_max + 1):
        for p in itertools.product(alphabet, repeat=k):
            all_kmers.add("".join(p))
    num_kmers = len(all_kmers)
    if verbose:
        logging.info(
            f"Generated {num_kmers:,} unique k-mers in {time.time() - t_start:.2f}s."
        )
    return sorted(list(all_kmers))


def _count_kmers_worker(args):
    """
    Worker function to count k-mers in a single sequence.
    The input sequence is expected to be in the target alphabet (e.g., binary '01').

    Args:
        args (tuple): A tuple containing:
            seq_id (str): The identifier for the sequence.
            sequence (str): The sequence string (e.g., binary string).
            k_min (int): Minimum k-mer length.
            k_max (int): Maximum k-mer length.
            valid_bases (str): String of valid characters for k-mers (e.g., '01').

    Returns:
        tuple: (seq_id, Counter object of k-mer counts).
    """
    seq_id, sequence, k_min, k_max, valid_bases = args
    kmer_counts = Counter()
    seq_len = len(sequence)
    # sequence.upper() is not needed if sequence is already binary ('01')

    for k in range(k_min, k_max + 1):
        if seq_len < k:  # Sequence is too short for this k
            continue
        for i in range(seq_len - k + 1):
            kmer = sequence[i : i + k]
            # Validate k-mer (optional if input sequence is guaranteed to be clean,
            # but good for robustness if binary conversion might produce unexpected chars
            # or if this function were to be used more generally).
            # For binary, this ensures no non-'01' characters slipped through.
            is_valid = True
            for base in kmer:
                if base not in valid_bases:
                    is_valid = False
                    break
            if is_valid:
                kmer_counts[kmer] += 1
    return seq_id, kmer_counts


def _get_checkpoint_base(fasta_files_list, k_min, k_max, alphabet_type="binary"):
    """
    Generates a base filename for checkpointing based on input files and parameters.
    Includes alphabet_type to distinguish between different k-mer approaches (e.g. ATGC vs binary).

    Args:
        fasta_files_list (list): List of FASTA file paths to process.
        k_min (int): Minimum k-mer length.
        k_max (int): Maximum k-mer length.
        alphabet_type (str): The type of k-mer encoding (e.g., "binary").

    Returns:
        str: MD5 hash string to use as a base filename for checkpoint files.
    """
    m = hashlib.md5()
    for f_path in sorted(fasta_files_list):
        m.update(os.path.abspath(f_path).encode())
    m.update(str(k_min).encode())
    m.update(str(k_max).encode())
    m.update(alphabet_type.encode())  # Add alphabet type to hash
    return m.hexdigest()


# --- Main Calculation Function ---
def calculate_kmer_profiles(
    input_path,
    k_min=3,
    k_max=12,
    num_workers=None,
    batch_size=10000,
    return_type="sparse_df",
    fasta_extensions=(".fa", ".fasta", ".fna", ".fas"),
    temp_dir=None,
    verbose=True,
    count_dtype=np.uint32,
    log_level="INFO",
    show_progress=True,
    quiet=False,
):
    """
    Calculates k-mer counts from FASTA files using a binary encoding (A/T->0, G/C->1)
    and batch processing to manage memory.

    Args:
        input_path (str): Path to a FASTA file or a directory containing FASTA files.
        k_min (int): Minimum k-mer length.
        k_max (int): Maximum k-mer length.
        num_workers (int, optional): Number of worker processes for parallelization.
                                     Defaults to the number of CPU cores.
        batch_size (int): Number of sequences to process in each batch.
        return_type (str): Desired format for the returned k-mer profiles.
                           Options: 'sparse_df', 'dense_df', 'sparse_matrix'.
        fasta_extensions (tuple): Tuple of file extensions to identify FASTA files
                                  when input_path is a directory.
        temp_dir (str, optional): Directory to store temporary batch files.
                                  If None, a system-default temporary directory is used.
        verbose (bool): If True, emit progress messages and warnings.
        count_dtype (numpy.dtype): NumPy data type for storing k-mer counts.
    log_level (str): Logging level to use. Defaults to "INFO".
    show_progress (bool): Whether to emit progress bars / milestone logging (auto-disables on non-TTY).

    Returns:
        Depending on `return_type`:
        - 'sparse_matrix': (scipy.sparse.csr_matrix, list of sequence IDs, list of k-mer names)
        - 'sparse_df': pandas.DataFrame (Sparse)
        - 'dense_df': pandas.DataFrame (Dense - WARNING: can consume a lot of memory)
        Returns None if 'dense_df' is requested and a MemoryError occurs.

    Raises:
        FileNotFoundError: If the input_path or FASTA files are not found.
        ValueError: For invalid parameters (e.g., k-mer range, return_type).
        RuntimeError: For critical errors during processing or saving.
        MemoryError: If memory limits are exceeded, especially with dense DataFrames.
    """
    # Set up logging for this function
    logger = setup_logging(log_level)

    # --- 1. Input Validation and File Handling ---
    if not os.path.exists(input_path):
        logger.error(f"Input path not found: {input_path}")
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pm = get_progress_manager(show_progress=show_progress, quiet=not verbose)
    fasta_files = []
    if os.path.isfile(input_path):
        fasta_files.append(input_path)
    elif os.path.isdir(input_path):
        if verbose:
            logger.info(f"Scanning directory: {input_path}")
        for ext in fasta_extensions:
            # Ensure correct pattern matching for case-insensitive extensions
            pattern_lower = os.path.join(input_path, f"*{ext.lower()}")
            pattern_upper = os.path.join(input_path, f"*{ext.upper()}")
            fasta_files.extend(glob.glob(pattern_lower))
            fasta_files.extend(glob.glob(pattern_upper))
        fasta_files = sorted(list(set(fasta_files)))  # Remove duplicates and sort
        if not fasta_files:
            logger.error(
                f"No files with extensions {fasta_extensions} found in {input_path}"
            )
            raise FileNotFoundError(
                f"No files with extensions {fasta_extensions} found in {input_path}"
            )
        if verbose:
            logger.info(f"Found {len(fasta_files)} FASTA file(s).")
    else:
        logger.error(f"Input path '{input_path}' is not a valid file or directory.")
        raise ValueError(f"Input path '{input_path}' is not a valid file or directory.")

    if not (1 <= k_min <= k_max):
        logger.error(
            f"Invalid k-mer range: k_min={k_min}, k_max={k_max}. k_min must be >= 1 and k_min <= k_max."
        )
        raise ValueError("Invalid k-mer range: k_min must be >= 1 and k_min <= k_max.")

    allowed_returns = ["sparse_df", "dense_df", "sparse_matrix"]
    if return_type not in allowed_returns:
        logger.error(
            f"Invalid return_type '{return_type}'. Choose from {allowed_returns}."
        )
        raise ValueError(
            f"Invalid return_type '{return_type}'. Choose from {allowed_returns}."
        )

    if num_workers is None:
        num_workers = cpu_count()
    if verbose:
        logger.info(
            f"Using {num_workers} worker processes. Batch size: {batch_size:,} sequences."
        )
        logger.info(f"Counting BINARY k-mers (A/T->0, G/C->1) for k={k_min}-{k_max}.")

    # Create progress manager and task for files (if multiple files)
    pm = get_progress_manager(show_progress=show_progress, quiet=quiet)
    files_progress = None
    if fasta_files and len(fasta_files) > 1:
        files_progress = pm.create_bar(
            total=len(fasta_files), desc="Scanning FASTA files"
        )

    # --- 2. Generate All Possible Binary K-mers (Done once) ---
    # For binary approach, the alphabet is '01'
    all_possible_kmers = _generate_all_kmers(
        k_min, k_max, alphabet="01", verbose=verbose
    )
    kmer_to_col_idx = {kmer: i for i, kmer in enumerate(all_possible_kmers)}
    num_total_kmers = len(all_possible_kmers)
    if num_total_kmers == 0:
        raise ValueError("No k-mers generated. Check k_min, k_max, and alphabet.")

    # --- 3. Setup Temporary Directory for Batches ---
    # Create a persistent temporary directory to avoid cleanup races
    import uuid

    batch_temp_dir = os.path.join(
        temp_dir or tempfile.gettempdir(), f"kmer_binary_batches_{uuid.uuid4().hex[:8]}"
    )
    os.makedirs(batch_temp_dir, exist_ok=True)

    if verbose:
        print(f"Using temporary directory for batches: {batch_temp_dir}")

    try:
        # --- 4. Process Sequences in Batches ---
        master_sequence_ids_ordered = []  # To store all sequence IDs in the order they are processed
        partial_matrix_files = []  # To store paths to saved .npz files for each batch
        batch_num = 0
        total_sequences_processed = 0
        global_processed_ids_check = set()  # To track unique IDs across all batches

        # Create progress bar for sequence processing
        sequences_progress = pm.create_bar(total=None, desc="Processing sequences")
        combine_progress = None

        pool = None  # Initialize pool to None
        final_sparse_matrix_csr = None
        try:
            # Create the multiprocessing pool
            pool = Pool(processes=num_workers)

            # Define a generator to yield sequences from all FASTA files
            def sequence_generator():
                for file_idx, file_path in enumerate(fasta_files):
                    file_basename = os.path.basename(file_path)
                    if verbose:
                        print(
                            f"Reading from file {file_idx + 1}/{len(fasta_files)}: {file_basename}"
                        )
                    try:
                        with open(file_path, "r") as handle:
                            for record in SeqIO.parse(handle, "fasta"):
                                yield (
                                    record,
                                    file_basename,
                                )  # Yield the record and its source filename
                    except Exception as e:
                        print(
                            f"Warning: Error reading records from {file_path}. Skipping this file. Error: {e}",
                            file=sys.stderr,
                        )
                    finally:
                        if files_progress is not None:
                            files_progress.update(1)

            seq_gen = sequence_generator()
            eof = False  # End of file/generator flag

            while not eof:
                batch_num += 1
                if verbose:
                    print(f"\nProcessing Batch {batch_num}...")
                start_time_batch = time.time()

                # Read sequences for the current batch
                current_batch_dna_sequences = []  # Store (unique_id, original_dna_sequence_string)
                current_batch_ids_ordered = []  # Store unique_ids for this batch in order

                for _ in range(batch_size):
                    try:
                        record, file_basename = next(seq_gen)

                        # Handle potentially duplicate sequence IDs
                        original_seq_id = record.id
                        current_seq_id = original_seq_id
                        dup_counter = 1
                        while current_seq_id in global_processed_ids_check:
                            if (
                                dup_counter == 1 and verbose
                            ):  # Print warning only on first detection
                                print(
                                    f"Warning: Duplicate ID '{original_seq_id}' from file '{file_basename}'. Renaming to avoid collision.",
                                    file=sys.stderr,
                                )
                            current_seq_id = (
                                f"{file_basename}_{original_seq_id}_dup{dup_counter}"
                            )
                            dup_counter += 1

                        if record.seq:  # Ensure sequence is not empty
                            current_batch_dna_sequences.append(
                                (current_seq_id, str(record.seq))
                            )
                            current_batch_ids_ordered.append(current_seq_id)
                            global_processed_ids_check.add(current_seq_id)
                            master_sequence_ids_ordered.append(current_seq_id)
                        elif verbose:
                            print(
                                f"Warning: Sequence '{original_seq_id}' in file '{file_basename}' is empty. Skipping.",
                                file=sys.stderr,
                            )

                    except StopIteration:
                        if files_progress is not None:
                            files_progress.close()
                            files_progress = None
                        eof = True  # Reached the end of the sequence generator
                        break  # Exit the inner loop for collecting batch sequences
                    except Exception as e:
                        # Catch other errors during record processing
                        print(
                            f"Warning: Error processing a record: {e}. Skipping this record.",
                            file=sys.stderr,
                        )

                if (
                    not current_batch_dna_sequences
                ):  # If the batch is empty (and not due to eof)
                    if verbose and not eof:
                        print(
                            "Batch is empty, but not at end of input. This might indicate an issue."
                        )
                    break  # Exit the outer while loop if no sequences were collected

                total_sequences_processed += len(current_batch_dna_sequences)
                sequences_progress.update(len(current_batch_dna_sequences))
                if verbose:
                    print(
                        f"  Read {len(current_batch_dna_sequences)} DNA sequences for batch {batch_num}."
                    )

                # Prepare tasks for parallel processing: Convert DNA to binary first
                batch_tasks = []
                for seq_id, dna_seq_str in current_batch_dna_sequences:
                    binary_seq_str = _dna_to_binary(dna_seq_str)  # Convert to binary
                    if (
                        not binary_seq_str and verbose
                    ):  # If binary string is empty (e.g. original DNA had no ATGC)
                        print(
                            f"Warning: Sequence ID '{seq_id}' resulted in an empty binary string. It will have no k-mers.",
                            file=sys.stderr,
                        )
                    # Pass '01' as valid_bases for the binary k-mer counting
                    batch_tasks.append((seq_id, binary_seq_str, k_min, k_max, "01"))

                if not batch_tasks:
                    if verbose:
                        print(
                            f"  No valid tasks for batch {batch_num} after binary conversion. Skipping matrix creation for this batch."
                        )
                    del current_batch_dna_sequences  # Clean up
                    continue  # Skip to next batch

                if verbose:
                    print(
                        f"  Starting binary k-mer counting for {len(batch_tasks)} sequences in batch {batch_num}..."
                    )
                batch_results = pool.map(
                    _count_kmers_worker, batch_tasks
                )  # List of (seq_id, Counter)

                # Build a sparse matrix for the current batch's results
                if verbose:
                    print(f"  Building sparse matrix for batch {batch_num}...")
                # DOK is efficient for incremental construction
                batch_matrix = dok_matrix(
                    (len(current_batch_ids_ordered), num_total_kmers), dtype=count_dtype
                )
                # Map sequence IDs of this batch to their row index in batch_matrix
                batch_id_to_row_idx = {
                    seq_id: i for i, seq_id in enumerate(current_batch_ids_ordered)
                }

                for seq_id, kmer_counts_for_seq in batch_results:
                    row_idx = batch_id_to_row_idx.get(seq_id)
                    if row_idx is None:  # Should not happen if logic is correct
                        if verbose:
                            print(
                                f"Warning: Sequence ID '{seq_id}' from worker results not found in batch ID list. Skipping.",
                                file=sys.stderr,
                            )
                        continue
                    for kmer, count in kmer_counts_for_seq.items():
                        col_idx = kmer_to_col_idx.get(kmer)
                        if (
                            col_idx is not None
                        ):  # K-mer is in our list of all possible k-mers
                            batch_matrix[row_idx, col_idx] = count
                        # Else: k-mer found by worker is not in all_possible_kmers (should not happen with '01' alphabet)

                batch_matrix_csr = (
                    batch_matrix.tocsr()
                )  # Convert to CSR for efficient storage and stacking

                # Save the partial matrix of this batch to a temporary file
                batch_file_path = os.path.join(
                    batch_temp_dir, f"batch_{batch_num:06d}.npz"
                )  # Padded for sorting
                if verbose:
                    print(
                        f"  Saving batch matrix (shape {batch_matrix_csr.shape}) to: {batch_file_path}"
                    )

                # Robust file saving with directory creation and atomic writes
                success = False
                for attempt in range(3):  # Try up to 3 times
                    try:
                        # Ensure the batch directory exists before each save attempt
                        os.makedirs(batch_temp_dir, exist_ok=True)

                        # Use atomic write: write to temporary file then rename
                        # Note: save_npz automatically adds .npz extension, so we use a temp name without .npz
                        temp_base = batch_file_path.replace(".npz", ".tmp")
                        temp_file_path = (
                            temp_base + ".npz"
                        )  # This will be created by save_npz

                        # save_npz automatically adds .npz, so we pass the base name
                        save_npz(temp_base, batch_matrix_csr)

                        # Atomic move from temp to final location
                        os.replace(temp_file_path, batch_file_path)

                        partial_matrix_files.append(batch_file_path)
                        success = True
                        break

                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            # If saving a batch fails after retries, it's a critical error as we can't combine later
                            raise RuntimeError(
                                f"Error saving batch matrix {batch_file_path} after {attempt + 1} attempts: {e}"
                            )
                        else:
                            if verbose:
                                print(f"  Attempt {attempt + 1} failed, retrying: {e}")
                            time.sleep(0.1 * (attempt + 1))  # Small backoff

                # Explicitly delete large objects to free memory for the next batch
                del (
                    batch_results,
                    batch_matrix,
                    batch_matrix_csr,
                    current_batch_dna_sequences,
                    batch_tasks,
                )
                if verbose:
                    print(
                        f"  Batch {batch_num} completed in {time.time() - start_time_batch:.2f}s."
                    )

            # --- 5. Combine Batch Results ---
            if not partial_matrix_files:
                if total_sequences_processed == 0:
                    raise RuntimeError(
                        "No valid sequences were processed from the input."
                    )
                else:  # This case should ideally not be reached if batches were processed
                    raise RuntimeError(
                        "Processing seemed to complete, but no partial matrix files were generated."
                    )

            if verbose:
                print(f"\nCombining {len(partial_matrix_files)} batch matrices...")
            start_time_combine = time.time()

            # Create progress bar for combining batches
            combine_progress = pm.create_bar(
                total=len(partial_matrix_files), desc="Loading batch matrices"
            )
            try:
                # Load all saved batch matrices
                matrices_to_stack = []
                for f in sorted(partial_matrix_files):  # Sort to ensure order
                    matrices_to_stack.append(load_npz(f))
                    combine_progress.update(1)

                # Validate that all matrices have the correct number of columns (k-mers)
                if (
                    not matrices_to_stack
                ):  # Should be caught by `if not partial_matrix_files`
                    raise ValueError("No batch matrices were loaded for stacking.")
                if not all(m.shape[1] == num_total_kmers for m in matrices_to_stack):
                    mismatched_shapes = [
                        (f, m.shape[1])
                        for f, m in zip(sorted(partial_matrix_files), matrices_to_stack)
                        if m.shape[1] != num_total_kmers
                    ]
                    raise ValueError(
                        f"Inconsistent number of columns in saved batch matrices. Expected {num_total_kmers}. Mismatches: {mismatched_shapes}"
                    )

                combine_progress.set_description("Stacking matrices")
                # Vertically stack the matrices
                final_sparse_matrix_csr = vstack(matrices_to_stack, format="csr")
            except MemoryError:
                raise MemoryError(
                    "Ran out of memory while trying to load and combine batch matrices. "
                    "Consider using a smaller --batch_size."
                )
            except Exception as e:
                raise RuntimeError(
                    f"An error occurred while combining batch matrices: {e}"
                )

            end_time_combine = time.time()
            if verbose:
                print(
                    f"Combined all batch matrices in {end_time_combine - start_time_combine:.2f}s."
                )
                print(f"Final sparse matrix shape: {final_sparse_matrix_csr.shape}")
                # Sanity check: rows in matrix should match total sequences processed
                if final_sparse_matrix_csr.shape[0] != len(master_sequence_ids_ordered):
                    print(
                        f"Warning: Final matrix row count ({final_sparse_matrix_csr.shape[0]}) "
                        f"differs from the count of master sequence IDs ({len(master_sequence_ids_ordered)}). "
                        "This might indicate an issue with ID tracking or batch processing.",
                        file=sys.stderr,
                    )
                elif final_sparse_matrix_csr.shape[0] != total_sequences_processed:
                    print(
                        f"Warning: Final matrix row count ({final_sparse_matrix_csr.shape[0]}) "
                        f"differs from total sequences processed ({total_sequences_processed}). Check logs.",
                        file=sys.stderr,
                    )

        except Exception as e:
            # Handle any errors in the inner processing block
            raise e

    finally:
        # Ensure the multiprocessing pool is properly closed and joined
        if pool:
            pool.close()
            pool.join()
            pool = None  # Mark as cleaned up

        # Clean up progress bars
        try:
            if sequences_progress:
                sequences_progress.close()
        except:
            pass
        try:
            if combine_progress:
                combine_progress.close()
        except:
            pass
        try:
            if files_progress:
                files_progress.close()
        except:
            pass

        # Clean up the temporary directory
        try:
            import shutil

            if os.path.exists(batch_temp_dir):
                shutil.rmtree(batch_temp_dir)
                if verbose:
                    print(f"Cleaned up temporary directory: {batch_temp_dir}")
        except Exception as cleanup_error:
            if verbose:
                print(
                    f"Warning: Failed to clean up temporary directory {batch_temp_dir}: {cleanup_error}"
                )

    # --- 6. Handle Return Type ---
    # `master_sequence_ids_ordered` contains the IDs for the rows of the final matrix
    # `all_possible_kmers` contains the names for the columns
    sequence_ids_ordered = master_sequence_ids_ordered

    if final_sparse_matrix_csr is None:
        raise RuntimeError("Final sparse matrix was not generated.")

    if return_type == "sparse_matrix":
        if verbose:
            print(
                "Returning final sparse matrix object and corresponding ID/k-mer mappings."
            )
        return final_sparse_matrix_csr, sequence_ids_ordered, all_possible_kmers
    elif return_type == "sparse_df":
        if verbose:
            print("Creating sparse Pandas DataFrame from the final matrix...")
        try:
            df = pd.DataFrame.sparse.from_spmatrix(
                final_sparse_matrix_csr,
                index=sequence_ids_ordered,
                columns=all_possible_kmers,
            )
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to create sparse Pandas DataFrame: {e}")
    elif return_type == "dense_df":
        if verbose:
            print(
                "Attempting to create dense Pandas DataFrame (WARNING: this can be memory-intensive)..."
            )
        # Heuristic: Warn if the dense matrix would be very large
        num_elements = (
            final_sparse_matrix_csr.shape[0] * final_sparse_matrix_csr.shape[1]
        )
        if num_elements > 10**8:  # Arbitrary threshold for large matrix
            print(
                f"Warning: The final matrix dimensions ({final_sparse_matrix_csr.shape[0]}x{final_sparse_matrix_csr.shape[1]}) "
                "are large. Converting to a dense DataFrame may cause a MemoryError.",
                file=sys.stderr,
            )
        try:
            # Convert CSR to dense array first, then to DataFrame
            dense_array = final_sparse_matrix_csr.toarray()
            df_dense = pd.DataFrame(
                dense_array, index=sequence_ids_ordered, columns=all_possible_kmers
            )
            if verbose:
                dense_mem_mb = df_dense.memory_usage(deep=True).sum() / (1024**2)
                print(f"  Dense DataFrame created. Memory usage: {dense_mem_mb:.2f} MB")
            return df_dense
        except MemoryError:
            if verbose:
                print(
                    "\nERROR: A MemoryError occurred while creating the dense Pandas DataFrame. "
                    "The dataset is too large for a dense representation in memory.",
                    file=sys.stderr,
                )
            return None  # Return None to indicate failure due to memory
        except Exception as e:
            raise RuntimeError(f"Failed to create dense Pandas DataFrame: {e}")


# --- Function to load saved k-mer results (if needed, e.g. for sparse format) ---
def load_kmer_results(base_path, verbose=True, log_level="INFO"):
    """
    Loads k-mer results that were saved in the sparse format
    (matrix.npz, rows.txt, cols.txt).

    Args:
        base_path (str): The base path used when saving (e.g., 'output/my_kmers').
                         The function will look for '{base_path}_sparse.npz', etc.
                         If using binary k-mers, ensure base_path includes the '_binary' tag
                         if it was added during saving (e.g., 'output/my_kmers_binary').
        verbose (bool): If True, log loading messages.
        log_level (str): Logging level to use. Defaults to "INFO".

    Returns:
        tuple: (scipy.sparse.csr_matrix, list of row names/IDs, list of col names/k-mers)
               Returns (None, None, None) if files are not found.
    """
    # Set up logging
    logger = setup_logging(log_level)

    # Determine file paths based on whether it's likely binary or ATGC output
    # This is a simple heuristic; a more robust way would be to pass the exact filenames
    # or have a metadata file.
    sparse_matrix_file = f"{base_path}_sparse.npz"
    rows_file = f"{base_path}_rows.txt"
    cols_file = f"{base_path}_cols.txt"

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
            if verbose:
                logger.info(f"Loading binary k-mer results from: {base_path}_binary*")
        else:
            if verbose:
                logger.error(
                    f"One or more result files not found for base path '{base_path}' (tried with and without '_binary' suffix)."
                )
            return None, None, None

    if verbose and sparse_matrix_file.startswith(base_path + "_binary"):
        pass  # Already logged above
    elif verbose:
        logger.info(f"Loading k-mer results from: {base_path}*")

    try:
        sparse_matrix = load_npz(sparse_matrix_file)
        with open(rows_file, "r") as f:
            row_names = [line.strip() for line in f]
        with open(cols_file, "r") as f:
            col_names = [line.strip() for line in f]
        if verbose:
            logger.info(
                f"Loaded sparse matrix ({sparse_matrix.shape}), {len(row_names)} row names, {len(col_names)} column names."
            )
        return sparse_matrix, row_names, col_names
    except Exception as e:
        if verbose:
            logger.error(f"Error loading k-mer results: {e}")
        return None, None, None


# --- Command-line Argument Parsing and Main Processing Function ---
def process_kmer_calculation(args):
    """
    Processes k-mer counts based on parsed arguments.

    This function encapsulates the main processing logic, making it importable
    by other scripts. It handles parsing file(s), counting k-mers, and saving results.

    Args:
        args: Parsed command-line arguments (can be from argparse or a similar object
              with the same attributes).

    Returns:
        Varies based on the specified return type:
        - If '--output' is specified, returns 0 on success or 1 on failure.
        - Otherwise, returns the k-mer profiles in the format specified by args.output_format.

    Raises:
        See exceptions from calculate_kmer_profiles().
    """
    # Resolve progress settings
    progress_settings = resolve_progress_settings(args)
    verbose = not progress_settings.get("quiet", False)
    logger = setup_logging(args.log_level if hasattr(args, "log_level") else "INFO")

    # Process FASTA extensions from comma-separated string to a tuple
    fasta_extensions = tuple(e.strip() for e in args.ext.split(",") if e.strip())
    if not fasta_extensions:
        logger.error(
            "No valid FASTA extensions provided. Use --ext (e.g., '.fa,.fasta')."
        )
        return 1

    # Map string dtype to NumPy dtype object
    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64}
    selected_dtype = dtype_map[args.dtype]

    if verbose:
        logger.info("--- Binary K-mer Counting Script ---")
        logger.info(f"Input: {args.input}")
        if args.output:
            logger.info(f"Output Base: {args.output}, Format: {args.output_format}")
        logger.info(f"K-mer range: {args.min_k}-{args.max_k}")
        logger.info(
            f"Workers: {args.workers if args.workers else cpu_count()}, Batch Size: {args.batch_size:,}"
        )
        logger.info(f"FASTA Extensions: {fasta_extensions}")
        logger.info(f"Count Data Type: {args.dtype}")
        logger.info("------------------------------------")

    try:
        # Determine the return_type needed for calculate_kmer_profiles based on saving preference
        # If saving dense (CSV or Parquet), we need a dense DataFrame from the function.
        # Otherwise, a sparse matrix is sufficient (and more memory-efficient).
        calc_return_type = (
            "dense_df"
            if args.output
            and (
                args.output_format == "dense_csv"
                or args.output_format == "dense_parquet"
            )
            else "sparse_matrix"
        )

        results = calculate_kmer_profiles(
            input_path=args.input,
            k_min=args.min_k,
            k_max=args.max_k,
            num_workers=args.workers,
            batch_size=args.batch_size,
            return_type=calc_return_type,
            fasta_extensions=fasta_extensions,
            temp_dir=args.temp_dir,
            verbose=verbose,
            count_dtype=selected_dtype,
            log_level=args.log_level if hasattr(args, "log_level") else "INFO",
            **progress_settings,  # Pass unified progress settings
        )

        # --- Handle Results and Saving (if --output is specified) ---
        if args.output:
            output_base = args.output
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_base)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                if verbose:
                    logger.info(f"Created output directory: {output_dir}")

            save_successful = True  # Flag to track if saving was successful

            # Handle case where dense DataFrame creation failed (results is None)
            if results is None and calc_return_type == "dense_df":
                if verbose:
                    logger.error(
                        "Dense DataFrame could not be created (likely due to MemoryError). "
                        "Cannot save in the requested dense format."
                    )
                save_successful = False

            if save_successful and results is not None:  # Proceed only if results exist
                # Add "_binary" tag to output filenames to distinguish from ATGC k-mer results
                output_base_tagged = f"{output_base}_binary"

                if args.output_format == "sparse":
                    if (
                        calc_return_type == "sparse_matrix"
                    ):  # results should be (matrix, rows, cols)
                        sparse_matrix, row_names, col_names = results
                        sparse_matrix_file = f"{output_base_tagged}_sparse.npz"
                        rows_file = f"{output_base_tagged}_rows.txt"
                        cols_file = f"{output_base_tagged}_cols.txt"
                        if verbose:
                            logger.info(
                                f"Saving sparse binary k-mer results to: {output_base_tagged}_*"
                            )
                        try:
                            save_npz(sparse_matrix_file, sparse_matrix)
                            with open(rows_file, "w") as f:
                                for item in row_names:
                                    f.write(f"{item}\n")
                            with open(cols_file, "w") as f:
                                for item in col_names:
                                    f.write(f"{item}\n")
                            if verbose:
                                logger.info(
                                    f"Outputs saved successfully:\n"
                                    f" - Matrix: {sparse_matrix_file}\n"
                                    f" - Row IDs: {rows_file}\n"
                                    f" - K-mers (Cols): {cols_file}"
                                )
                        except Exception as e:
                            logger.error(f"Error saving sparse files: {e}")
                            save_successful = False
                    else:  # Should not happen if calc_return_type logic is correct
                        logger.error(
                            f"Cannot save in sparse format. Expected 'sparse_matrix' results, "
                            f"but received type {type(results)}."
                        )
                        save_successful = False

                elif args.output_format == "dense_csv":
                    if calc_return_type == "dense_df" and isinstance(
                        results, pd.DataFrame
                    ):
                        df_dense = results
                        csv_file = f"{output_base_tagged}_dense.csv"
                        if verbose:
                            logger.info(
                                f"Saving dense binary k-mer DataFrame to CSV: {csv_file}"
                            )
                        try:
                            df_dense.to_csv(csv_file, index=True)
                            if verbose:
                                logger.info(
                                    f"Dense CSV saved successfully to {csv_file}."
                                )
                        except Exception as e:
                            logger.error(f"Error saving dense CSV: {e}")
                            save_successful = False
                    else:
                        logger.error(
                            f"Cannot save as dense CSV. Expected a pandas DataFrame, "
                            f"but received type {type(results)}."
                        )
                        save_successful = False

                elif args.output_format == "dense_parquet":
                    if calc_return_type == "dense_df" and isinstance(
                        results, pd.DataFrame
                    ):
                        df_dense = results
                        parquet_file = f"{output_base_tagged}_dense.parquet"
                        if verbose:
                            logger.info(
                                f"Saving dense binary k-mer DataFrame to Parquet: {parquet_file}"
                            )
                        try:
                            df_dense.to_parquet(
                                parquet_file,
                                index=True,
                                engine="pyarrow",
                                compression="zstd",
                            )
                            if verbose:
                                logger.info(
                                    f"Dense Parquet saved successfully to {parquet_file}."
                                )
                        except ImportError:
                            logger.error(
                                "The 'pyarrow' library is required to save in Parquet format. "
                                "Please install it (e.g., 'pip install pyarrow')."
                            )
                            save_successful = False
                        except Exception as e:
                            logger.error(f"Error saving dense Parquet: {e}")
                            save_successful = False
                    else:
                        logger.error(
                            f"Cannot save as dense Parquet. Expected a pandas DataFrame, "
                            f"but received type {type(results)}."
                        )
                        save_successful = False
            elif (
                results is None and not save_successful
            ):  # Already handled above, just pass
                pass
            elif (
                results is None
            ):  # General case if results is None and not caught above
                if verbose:
                    logger.error("No results were generated to save.")
                save_successful = False

            if not save_successful:
                if verbose:
                    logger.error("One or more output saving steps failed.")
                return 1  # Exit with error if any saving operation failed

            # Success case for saving
            return 0

        elif verbose:  # --output not specified
            if results is None:
                logger.warning(
                    "Calculation completed, but no results were generated (e.g., MemoryError for dense DataFrame). "
                    "Results not saved as --output was not specified."
                )
            else:
                logger.info(
                    "Calculation complete. Results were not saved as --output was not specified."
                )

            # Return the results directly
            return results

    except (FileNotFoundError, ValueError, RuntimeError, IOError, MemoryError) as e:
        logger.error(f"An ERROR occurred: {e}")
        return 1
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An UNEXPECTED ERROR occurred: {e}")
        import traceback

        traceback.print_exc()  # Print full traceback for unexpected errors
        return 1

    if verbose:
        logger.info("Script finished successfully.")
    return 0  # Default success return code


# --- Command-line Argument Parsing and Main Execution ---
def parse_arguments():
    """
    Parses command-line arguments for k-mer calculation.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        description="Count k-mers from FASTA file(s) using a BINARY encoding (A/T->0, G/C->1) and batch processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show default values in help
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input FASTA file or a directory containing FASTA files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="Base path for saving the final output files. If not provided, results are returned but not saved.",
    )
    parser.add_argument(
        "--ext",
        default=".fa,.fasta,.fna,.fas",
        help="Comma-separated list of FASTA file extensions to look for if --input is a directory.",
    )
    parser.add_argument("--min_k", type=int, default=3, help="Minimum k-mer length.")
    parser.add_argument("--max_k", type=int, default=12, help="Maximum k-mer length.")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Defaults to the number of available CPU cores.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Number of sequences to process per batch. Adjust to control memory usage.",
    )
    parser.add_argument(
        "--temp_dir",
        default=None,
        help="Directory for storing temporary batch files. If None, a system default is used.",
    )
    parser.add_argument(
        "--output_format",
        choices=["sparse", "dense_csv", "dense_parquet"],
        default="sparse",
        help=(
            "Format for saving output if --output is specified. "
            "'sparse' saves a .npz matrix file and .txt files for row/column names. "
            "'dense_csv' saves a CSV file (can be very large). "
            "'dense_parquet' saves a Parquet file (efficient, requires 'pyarrow')."
        ),
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress progress messages and warnings.",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32", "uint64"],
        default="uint32",
        help="NumPy data type for k-mer counts in the sparse matrix. Affects memory and max count.",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level.",
    )
    # Add unified progress arguments (excluding --quiet since it already exists)
    update_cli_args(parser, add_quiet=False)

    args = parser.parse_args()
    return args


def main():
    """Main entry point for the k-mer calculation CLI."""
    args = parse_arguments()
    return process_kmer_calculation(args)


if __name__ == "__main__":
    sys.exit(main())
