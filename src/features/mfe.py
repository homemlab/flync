import pandas as pd
import sys
import argparse
import os # Import os to work with file paths, extensions, and CPU count
import multiprocessing # Import multiprocessing for parallel execution
import math # For ceiling function
import logging # Added for standardized logging

# --- Logging Setup ---
def setup_logging(log_level):
    """
    Configures the root logger for the application.

    Args:
        log_level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        # Default to INFO if an invalid log level string is provided
        logging.warning(f"Invalid log level '{log_level}'. Defaulting to INFO.")
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level, 
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        stream=sys.stdout # Direct logs to stdout
    )
    # For warnings/errors from other libraries that use logging, ensure they are also visible
    # if our level is DEBUG, for example. This is generally handled by basicConfig level.

# --- Dependency Check ---
# Check specifically for PyArrow
pyarrow_available = False
try:
    import pyarrow
    pyarrow_available = True
    # Use logging instead of print for initial messages, once logging is configured.
    # logging.info("Using 'pyarrow' engine for file operations where possible.") 
except ImportError:
    # logging.warning("'pyarrow' package not found. Install with: pip install pandas pyarrow")
    # logging.warning("Parquet file support (.parquet) will be unavailable.")
    pass # Warnings will be emitted by functions that need it, or at CLI start.

# --- ViennaRNA Check ---
try:
    import RNA
except ImportError:
    # These are critical, so print to stderr and exit early if logging isn't set up yet.
    print("CRITICAL ERROR: The 'ViennaRNA' package is not installed or cannot be found.", file=sys.stderr)
    print("Please install it using 'pip install ViennaRNA' and ensure the ViennaRNA suite is installed on your system.", file=sys.stderr)
    print("See: https://www.tbi.univie.ac.at/RNA/#download", file=sys.stderr)
    sys.exit(1)

# --- Helper function for multiprocessing ---
def get_folding_results_mp_helper_indexed(item: tuple[int, str]) -> tuple[int, float | None, str | None]:
    """
    Calculates Minimum Free Energy (MFE) and RNA secondary structure for a single RNA sequence.

    This function is designed to be used as a worker in a multiprocessing pool.
    It takes a tuple containing an original index and a sequence, performs RNA folding
    using ViennaRNA, and returns the index along with the MFE and structure.

    Args:
        item (tuple[int, str]): A tuple where:
            - item[0] (int): The original index of the sequence within its current processing batch.
            - item[1] (str): The RNA sequence string (e.g., "ACGUG...").

    Returns:
        tuple[int, float | None, str | None]: A tuple containing:
            - original_index (int): The input original index.
            - mfe (float | None): The calculated MFE. pd.NA if calculation fails or sequence is invalid.
            - structure (str | None): The predicted secondary structure in dot-bracket notation. 
                                      pd.NA if calculation fails, sequence is invalid, or structure not requested.
    """
    original_idx, sequence = item
    if not isinstance(sequence, str) or not sequence:
        logging.debug(f"Sequence at index {original_idx} is invalid (empty or not a string). Skipping.")
        return original_idx, pd.NA, pd.NA
    try:
        structure, mfe = RNA.fold(sequence)
        logging.debug(f"Successfully folded sequence at index {original_idx} (len {len(sequence)}): MFE={mfe}")
        return original_idx, float(mfe), str(structure)
    except Exception as e:
        logging.warning(f"RNA folding failed for sequence at index {original_idx} (len {len(sequence)}): {e}")
        return original_idx, pd.NA, pd.NA


def calculate_all_mfe_and_structure(
    df_to_process: pd.DataFrame, 
    sequence_col: str = 'Sequence',
    include_structure: bool = False,
    num_processes: int | None = None
) -> pd.DataFrame:
    """
    Calculates MFE and optionally the secondary structure for sequences in a DataFrame subset.

    Uses multiprocessing to parallelize RNA folding calculations.

    Args:
        df_to_process (pd.DataFrame): DataFrame containing sequences that require MFE/structure calculation.
                                      Must have the `sequence_col`.
        sequence_col (str): Name of the column in `df_to_process` that contains RNA sequences.
        include_structure (bool): If True, the secondary structure (dot-bracket notation)
                                  is calculated and included in the output. Defaults to False.
        num_processes (int, optional): Number of worker processes for multiprocessing.
                                       Defaults to `os.cpu_count()`.

    Returns:
        pd.DataFrame: A DataFrame with the same index as `df_to_process`. It includes the
                      original data from `df_to_process` plus new columns:
                      'MFE' (float, or pd.NA on failure) and, if `include_structure` is True,
                      'Structure' (str, or pd.NA on failure).

    Raises:
        KeyError: If `sequence_col` is not found in `df_to_process`.
    """
    if sequence_col not in df_to_process.columns:
        logging.error(f"Sequence column '{sequence_col}' not found in the DataFrame subset for calculation.")
        raise KeyError(f"Column '{sequence_col}' not found in the DataFrame subset provided for calculation.")

    # Enumerate sequences from the df_to_process. Indices are local to this df.
    sequences_with_local_indices = list(enumerate(df_to_process[sequence_col].tolist()))
    total_sequences_to_calculate = len(sequences_with_local_indices)

    if total_sequences_to_calculate == 0:
        logging.info("No new sequences to calculate in this run.")
        # Return a copy of the input df with empty MFE/Structure columns if they don't exist
        df_out = df_to_process.copy()
        if 'MFE' not in df_out.columns:
            df_out['MFE'] = pd.NA
        if include_structure and 'Structure' not in df_out.columns:
            df_out['Structure'] = pd.NA
        return df_out

    if num_processes is None:
        num_processes = os.cpu_count()
    logging.info(f"Calculating MFE for {total_sequences_to_calculate} sequences using {num_processes} processes...")

    # Pre-allocate lists for results, sized for df_to_process
    mfe_values = [pd.NA] * total_sequences_to_calculate
    structure_values = [pd.NA] * total_sequences_to_calculate if include_structure else None
    processed_count = 0

    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_processes)
        # Use imap_unordered for potentially better performance with uneven task times
        results_iterator = pool.imap_unordered(get_folding_results_mp_helper_indexed, sequences_with_local_indices)

        for local_idx, mfe, struct in results_iterator:
            mfe_values[local_idx] = mfe
            if include_structure and structure_values is not None: # Ensure structure_values was initialized
                structure_values[local_idx] = struct

            processed_count += 1
            if processed_count % 100 == 0 or processed_count == total_sequences_to_calculate:
                progress = (processed_count / total_sequences_to_calculate) * 100
                # Use logging for progress, consider if this is too verbose for INFO
                # logging.debug(f"Calculation Progress: {processed_count}/{total_sequences_to_calculate} ({progress:.1f}%) processed...")
                # For CLI, print might be better for \\r updates
                print(f"\\rCalculation Progress: {processed_count}/{total_sequences_to_calculate} ({progress:.1f}%) processed...", end="")

    except Exception as e:
         logging.error(f"Error during multiprocessing MFE calculation: {e}", exc_info=True)
         # Indicate failure by preparing a DataFrame with NA columns
         df_out = df_to_process.copy()
         df_out['MFE'] = pd.NA
         if include_structure:
             df_out['Structure'] = pd.NA
         # Ensure pool is closed even on error before returning
         if pool:
             pool.close()
             pool.join()
         if total_sequences_to_calculate > 0: 
             print() # Final newline if progress was printed
         return df_out # Propagate failure state

    finally:
        if pool:
            pool.close()
            pool.join()
        if total_sequences_to_calculate > 0: # Avoid printing newline if no calculations were done
            print() # Final newline after progress indicator

    logging.info("MFE calculations for the current set of sequences complete.")

    # Assign results to a copy of df_to_process
    df_out = df_to_process.copy() # df_out will have the same index as df_to_process
    df_out['MFE'] = mfe_values
    if include_structure and structure_values is not None:
        df_out['Structure'] = structure_values

    df_out['MFE'] = pd.to_numeric(df_out['MFE'], errors='coerce')
    return df_out


def load_checkpoint_batch(filename: str, include_structure: bool, sequence_col: str) -> pd.DataFrame | None:
    """
    Attempts to load and validate a batch file as a checkpoint.

    Supports CSV, TSV, and Parquet formats (Parquet requires pyarrow).
    Validates that the checkpoint contains 'MFE' and, if expected, 'Structure' columns,
    as well as the specified sequence column.

    Args:
        filename (str): Path to the checkpoint file.
        include_structure (bool): Whether the 'Structure' column is expected in the checkpoint.
        sequence_col (str): The name of the sequence column expected in the checkpoint.

    Returns:
        pd.DataFrame | None: The loaded DataFrame if successful and valid, otherwise None.
    """
    try:
        _, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lower()
        df_checkpoint = None
        csv_engine = 'pyarrow' if pyarrow_available else None # Use pyarrow for CSV if available

        logging.debug(f"Attempting to load checkpoint file: {filename}")
        if file_ext == '.csv':
            df_checkpoint = pd.read_csv(filename, sep=',', engine=csv_engine)
        elif file_ext == '.tsv':
            df_checkpoint = pd.read_csv(filename, sep='\t', engine=csv_engine)
        elif file_ext == '.parquet':
            if not pyarrow_available:
                logging.warning(f"Cannot load Parquet checkpoint '{filename}', pyarrow not available.")
                return None
            df_checkpoint = pd.read_parquet(filename, engine='pyarrow')
        else:
            logging.warning(f"Unknown extension for checkpoint file '{filename}'. Cannot load.")
            return None

        # Validate checkpoint
        if 'MFE' not in df_checkpoint.columns:
            logging.warning(f"Checkpoint file '{filename}' is missing 'MFE' column. Will re-process corresponding data.")
            return None
        if include_structure and 'Structure' not in df_checkpoint.columns:
            logging.warning(f"Checkpoint file '{filename}' is missing 'Structure' column (when expected). Will re-process.")
            return None
        if sequence_col not in df_checkpoint.columns:
            logging.warning(f"Checkpoint file '{filename}' is missing sequence column '{sequence_col}'. May indicate mismatch. Will re-process.")
            return None
        
        logging.info(f"Successfully loaded and validated checkpoint: {filename}")
        return df_checkpoint
    except Exception as e:
        logging.warning(f"Could not load or validate checkpoint file '{filename}': {e}. Will re-process.")
        return None


def write_output_file(df: pd.DataFrame, filename: str):
    """
    Writes a DataFrame to a file, inferring format from extension (CSV, TSV, Parquet).

    Args:
        df (pd.DataFrame): The DataFrame to write.
        filename (str): The path to the output file.

    Raises:
        SystemExit: If an unsupported output format is specified or pyarrow is needed but unavailable.
        SystemExit: If an error occurs during file writing.
    """
    try:
        _, output_ext = os.path.splitext(filename)
        output_ext = output_ext.lower()

        logging.info(f"Writing output file: {filename} ({len(df)} rows)")

        if output_ext == '.csv':
            df.to_csv(filename, sep=',', index=False)
        elif output_ext == '.tsv':
            df.to_csv(filename, sep='\t', index=False)
        elif output_ext == '.parquet':
            if not pyarrow_available:
                 logging.error(f"Cannot write Parquet file '{filename}' because 'pyarrow' package is not installed.")
                 sys.exit(1) # Critical error for specified output type
            df.to_parquet(filename, engine='pyarrow', index=False)
        else:
            logging.error(f"Unsupported output file format '{output_ext}' for file '{filename}'. Please use .csv, .tsv, or .parquet.")
            sys.exit(1) # Critical error for specified output type
        logging.info(f"File saved successfully: {filename}")
    except Exception as e:
        logging.error(f"Error writing output file {filename}: {e}", exc_info=True)
        sys.exit(1) # Critical error during write


def process_mfe_calculations(
    input_file: str,
    output_file: str,
    sequence_col: str = "Sequence",
    include_structure: bool = False,
    batch_size: int = 0,
    num_processes: int | None = None
):
    """
    Core logic for calculating MFE and optionally structure for sequences from an input file.
    Supports batching and checkpointing.

    Args:
        input_file (str): Path to the input file (.csv, .tsv, .parquet).
        output_file (str): Path/base name for the output file(s).
        sequence_col (str): Name of the sequence column in the input file.
        include_structure (bool): If True, include secondary structure in the output.
        batch_size (int): Size for batch processing. 0 means single output file (no batching).
                          Enables checkpointing if > 0.
        num_processes (int, optional): Number of cores for multiprocessing. Defaults to os.cpu_count().

    Returns:
        pd.DataFrame: The final DataFrame containing all results.
                      This DataFrame is also written to file(s) as a side effect.

    Raises:
        SystemExit: For critical errors like file not found, unsupported format, or write errors.
        KeyError: If the sequence column is not found in the input data.
    """
    if batch_size < 0:
        logging.error("batch_size cannot be negative.")
        sys.exit(1)

    # --- Read Full Input DataFrame ---
    try:
        _, input_ext = os.path.splitext(input_file)
        input_ext = input_ext.lower()
        logging.info(f"Reading input file: {input_file}")
        csv_engine = 'pyarrow' if pyarrow_available else None

        if input_ext == '.csv':
            full_input_df = pd.read_csv(input_file, sep=',', engine=csv_engine)
        elif input_ext == '.tsv':
            full_input_df = pd.read_csv(input_file, sep='\t', engine=csv_engine)
        elif input_ext == '.parquet':
            if not pyarrow_available:
                 logging.error("Cannot read Parquet file '{input_file}', 'pyarrow' not installed.")
                 sys.exit(1)
            full_input_df = pd.read_parquet(input_file, engine='pyarrow')
        else:
            logging.error(f"Unsupported input file format '{input_ext}' for file '{input_file}'.")
            sys.exit(1)
        total_rows = len(full_input_df)
        logging.info(f"Successfully read {total_rows} total rows from {input_file}.")
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error reading input file {input_file}: {e}", exc_info=True)
        sys.exit(1)

    if total_rows == 0:
        logging.info("Input file is empty. No processing needed.")
        # Write an empty output file consistent with expected columns
        empty_df_cols = list(full_input_df.columns) + ['MFE']
        if include_structure:
            empty_df_cols.append('Structure')
        empty_df = pd.DataFrame(columns=empty_df_cols)
        write_output_file(empty_df, output_file) # Handles single or batch naming
        return empty_df

    if sequence_col not in full_input_df.columns:
        logging.error(f"Sequence column '{sequence_col}' not found in input file '{input_file}'.")
        raise KeyError(f"Sequence column '{sequence_col}' not found.")

    # --- Initialize final DataFrame that will hold all results ---
    final_results_df = full_input_df.copy()
    final_results_df['MFE'] = pd.NA
    if include_structure:
        final_results_df['Structure'] = pd.NA

    # --- Checkpointing and Identifying Rows for Calculation (Batch Mode Only) ---
    rows_needing_calculation_indices = list(range(total_rows)) # Initially, all rows

    if batch_size > 0:
        logging.info("--- Checkpoint Scan (Batch Mode) ---")
        num_expected_batches = math.ceil(total_rows / batch_size)
        output_base, output_ext_for_file = os.path.splitext(output_file)
        
        processed_rows_from_checkpoints = 0

        for i in range(num_expected_batches):
            batch_start_orig_idx = i * batch_size
            batch_end_orig_idx = min((i + 1) * batch_size, total_rows)
            
            batch_start_num_fn = batch_start_orig_idx + 1
            batch_end_num_fn = batch_end_orig_idx

            batch_filename = f"{output_base}_{batch_start_num_fn}-{batch_end_num_fn}{output_ext_for_file}"

            if os.path.exists(batch_filename):
                logging.info(f"Found potential checkpoint file: {batch_filename}")
                df_checkpoint = load_checkpoint_batch(batch_filename, include_structure, sequence_col)
                if df_checkpoint is not None:
                    expected_batch_len = batch_end_orig_idx - batch_start_orig_idx
                    if len(df_checkpoint) == expected_batch_len:
                        logging.info(f"Valid checkpoint loaded for batch {i+1} (rows {batch_start_num_fn}-{batch_end_num_fn}).")
                        for j in range(len(df_checkpoint)):
                            original_row_idx = batch_start_orig_idx + j
                            final_results_df.loc[original_row_idx, 'MFE'] = df_checkpoint.iloc[j]['MFE']
                            if include_structure:
                                final_results_df.loc[original_row_idx, 'Structure'] = df_checkpoint.iloc[j]['Structure']
                        
                        for k_idx in range(batch_start_orig_idx, batch_end_orig_idx):
                            if k_idx in rows_needing_calculation_indices:
                                rows_needing_calculation_indices.remove(k_idx)
                        processed_rows_from_checkpoints += len(df_checkpoint)
                    else:
                        logging.warning(f"Checkpoint file '{batch_filename}' has {len(df_checkpoint)} rows, expected {expected_batch_len}. Will re-process this batch's rows if not covered by other valid checkpoints.")
        
        if processed_rows_from_checkpoints > 0:
            logging.info(f"{processed_rows_from_checkpoints} rows' data loaded from existing checkpoints.")
        logging.info(f"{len(rows_needing_calculation_indices)} rows will be processed/re-processed.")
        logging.info("--- End Checkpoint Scan ---")

    # --- Perform Calculations for Remaining Rows ---
    if rows_needing_calculation_indices:
        df_to_calculate = full_input_df.iloc[rows_needing_calculation_indices].copy()
        try:
            calculated_data_df = calculate_all_mfe_and_structure(
                df_to_calculate,
                sequence_col=sequence_col,
                include_structure=include_structure,
                num_processes=num_processes
            )
            # Merge calculated data back into final_results_df using original indices
            for original_idx in rows_needing_calculation_indices:
                if original_idx in calculated_data_df.index:
                    final_results_df.loc[original_idx, 'MFE'] = calculated_data_df.loc[original_idx, 'MFE']
                    if include_structure:
                        final_results_df.loc[original_idx, 'Structure'] = calculated_data_df.loc[original_idx, 'Structure']
        except KeyError as e: 
            logging.error(f"Error during calculation setup (likely missing sequence column '{sequence_col}'): {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred during main calculation phase: {e}", exc_info=True)
            sys.exit(1)
    else:
        logging.info("No new rows to calculate. All data potentially loaded from checkpoints or input was empty.")

    # --- Write Output (Batch or Single File) using the fully assembled final_results_df ---
    if batch_size > 0:
        logging.info(f"--- Writing Output in Batches ({batch_size} rows/batch) ---")
        num_output_batches = math.ceil(total_rows / batch_size)
        output_base, output_ext_for_file = os.path.splitext(output_file)

        for i in range(num_output_batches):
            start_row_idx = i * batch_size
            end_row_idx = min((i + 1) * batch_size, total_rows)
            
            start_row_num_fn = start_row_idx + 1
            end_row_num_fn = end_row_idx
            
            logging.info(f"Preparing batch {i+1}/{num_output_batches} for writing (Original Rows {start_row_num_fn}-{end_row_num_fn})")
            df_chunk_to_write = final_results_df.iloc[start_row_idx:end_row_idx]
            batch_filename = f"{output_base}_{start_row_num_fn}-{end_row_num_fn}{output_ext_for_file}"
            write_output_file(df_chunk_to_write, batch_filename)
        logging.info("--- All batches written. ---")
    else: 
        logging.info("--- Writing Output to Single File ---")
        write_output_file(final_results_df, output_file)

    logging.info("MFE processing complete.")
    return final_results_df


def main():
    """Command-line interface for MFE calculation script."""
    parser = argparse.ArgumentParser(
        description="Calculate MFE and optionally RNA secondary structure for sequences in a file. "
                    "Supports CSV, TSV, and Parquet input/output. Enables checkpointing for batch mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", help="Path to the input file (.csv, .tsv, .parquet) containing sequences.")
    parser.add_argument("output_file", help="Path/base name for the output file(s). Extension determines format.")
    parser.add_argument("-s", "--sequence_col", default="Sequence", 
                        help="Name of the column containing RNA sequences.")
    parser.add_argument("--include_structure", action="store_true", 
                        help="Include the dot-bracket RNA secondary structure in the output.")
    parser.add_argument("-b", "--batch_size", type=int, default=0, 
                        help="Batch size for processing and output files. If 0, a single output file is created. "
                             "A non-zero value enables checkpointing, where completed batch files are not re-processed.")
    parser.add_argument("--num_processes", "-p", type=int, default=None,
                        help="Number of parallel processes to use for calculations. Defaults to all available CPU cores.")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set the logging level.")

    args = parser.parse_args()

    setup_logging(args.log_level)

    # PyArrow availability message after logging is set up
    if pyarrow_available:
        logging.info("'pyarrow' package found. Will be used for Parquet files and optionally for CSV/TSV I/O.")
    else:
        logging.warning("'pyarrow' package not found. Parquet file support (.parquet) will be unavailable. CSV/TSV I/O might be slower.")
        logging.warning("Install with: pip install pandas pyarrow")

    try:
        process_mfe_calculations(
            input_file=args.input_file,
            output_file=args.output_file,
            sequence_col=args.sequence_col,
            include_structure=args.include_structure,
            batch_size=args.batch_size,
            num_processes=args.num_processes
        )
        logging.info("Script finished successfully.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        logging.error(f"Column-related error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)

# --- Run Main Function ---    
if __name__ == "__main__":
    # freeze_support() is necessary for Windows when using multiprocessing
    # and creating executables (e.g. with PyInstaller).
    multiprocessing.freeze_support() 
    main()
