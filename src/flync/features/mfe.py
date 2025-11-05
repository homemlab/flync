import argparse
import logging
import math
import multiprocessing
import os
import re
import subprocess
import sys
from typing import Optional, Tuple, Union

import pandas as pd

# Import progress utilities
try:
    from src.utils.progress import get_progress_manager, update_cli_args, resolve_progress_settings
except ImportError:
    # Try relative import when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.progress import get_progress_manager, update_cli_args, resolve_progress_settings


# --- Logging Setup ---
def setup_logging(log_level):
    """
    Configures the root logger for the application.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level '{log_level}'. Defaulting to INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        stream=sys.stdout,
    )


# --- Dependency Check ---
pyarrow_available = False
try:
    import pyarrow

    pyarrow_available = True
except ImportError:
    pass


# --- LinearFold Prediction Function ---
def get_linearfold_prediction(sequence: str) -> Tuple[str, float]:
    """
    Predicts the secondary structure and calculates the Minimum Free Energy (MFE)
    of an RNA sequence using LinearFold.

    Args:
        sequence: The nucleotide sequence.

    Returns:
        A tuple containing:
        - The predicted secondary structure in dot-bracket notation (str).
        - The calculated MFE value as a float (kcal/mol).

    Raises:
        RuntimeError: If the LinearFold command fails to execute.
        ValueError: If the output from LinearFold cannot be parsed correctly.
    """
    # Ensure the sequence is a single line without invalid characters for the command
    if not re.match(r"^[ATGCUatgcuNn]+$", sequence):
        raise ValueError("Sequence contains invalid characters.")
    sequence = sequence.strip().upper().replace("T", "U")  # Normalize to uppercase for RNA folding

    try:
        # Execute the LinearFold command
        process = subprocess.run(
            ["/home/chlab/LinearFold/linearfold", "-V"],
            input=sequence,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,  # This will raise CalledProcessError for non-zero exit codes
        )
    except FileNotFoundError:
        raise RuntimeError("LinearFold command not found. Is it installed and in your PATH?")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LinearFold encountered an error: {e.stderr}")

    # The standard output from LinearFold contains the results.
    # Example output:
    # GGUGCUGAUGAUGUGAGUUGGUUAGUAUUUGUCUGAUUGAAU
    # .((((....)))).((.((...)).))............... (-5.40)
    output_lines = process.stdout.strip().split("\n")

    # We expect at least 2 lines in the output: sequence and structure+MFE
    if len(output_lines) < 2:
        raise ValueError(f"Unexpected output format from LinearFold: {process.stdout}")

    # The last line contains the structure and the MFE
    prediction_line = output_lines[-1]

    # The dot-bracket structure is the first part of the line
    structure = prediction_line.split(" ")[0]

    # The MFE value is enclosed in parentheses at the end of the line
    mfe_match = re.search(r"\((.*?)\)", prediction_line.split(" ")[-1])

    if not structure or not mfe_match:
        raise ValueError(f"Could not parse structure or MFE from output: '{prediction_line}'")

    mfe_value = float(mfe_match.group(1))

    return structure, mfe_value


# --- Helper function for multiprocessing ---
def get_folding_results_mp_helper_indexed(
    item: Tuple[int, str],
) -> Tuple[int, Union[float, None], Union[str, None]]:
    """
    Calculates Minimum Free Energy (MFE) and RNA secondary structure for a single RNA sequence using LinearFold.

    This function is designed to be used as a worker in a multiprocessing pool.
    It takes a tuple containing an original index and a sequence, performs RNA folding,
    and returns the index along with the MFE and structure.

    Args:
        item (tuple[int, str]): A tuple where:
            - item[0] (int): The original index of the sequence.
            - item[1] (str): The RNA sequence string.

    Returns:
        tuple[int, float | None, str | None]: A tuple containing the original index, MFE, and structure.
                                              Returns pd.NA for MFE/structure on failure.
    """
    original_idx, sequence = item
    if not isinstance(sequence, str) or not sequence:
        logging.debug(f"Sequence at index {original_idx} is invalid. Skipping.")
        return original_idx, pd.NA, pd.NA
    try:
        # Use the LinearFold function
        structure, mfe = get_linearfold_prediction(sequence)
        logging.debug(
            f"Successfully folded sequence at index {original_idx} (len {len(sequence)}): MFE={mfe}"
        )
        return original_idx, float(mfe), str(structure)
    except (RuntimeError, ValueError) as e:
        logging.warning(
            f"LinearFold folding failed for sequence at index {original_idx} (len {len(sequence)}): {e}"
        )
        return original_idx, pd.NA, pd.NA
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in worker for sequence at index {original_idx}: {e}"
        )
        return original_idx, pd.NA, pd.NA


def calculate_all_mfe_and_structure(
    df_to_process: pd.DataFrame,
    sequence_col: str = "Sequence",
    include_structure: bool = False,
    num_processes: Union[int, None] = None,
    show_progress: bool = True,
    quiet: bool = False,
) -> pd.DataFrame:
    """
    Calculates MFE and optionally the secondary structure for sequences in a DataFrame using LinearFold.

    Uses multiprocessing to parallelize RNA folding calculations.
    """
    if sequence_col not in df_to_process.columns:
        logging.error(f"Sequence column '{sequence_col}' not found in the DataFrame.")
        raise KeyError(f"Column '{sequence_col}' not found.")

    sequences_with_local_indices = list(enumerate(df_to_process[sequence_col].tolist()))
    total_sequences_to_calculate = len(sequences_with_local_indices)

    if total_sequences_to_calculate == 0:
        logging.info("No new sequences to calculate.")
        df_out = df_to_process.copy()
        if "mfe" not in df_out.columns:
            df_out["mfe"] = pd.NA
        if include_structure and "structure" not in df_out.columns:
            df_out["structure"] = pd.NA
        return df_out

    if num_processes is None:
        num_processes = os.cpu_count()
    logging.info(
        f"Calculating MFE for {total_sequences_to_calculate} sequences using {num_processes} processes with LinearFold..."
    )

    mfe_values = [pd.NA] * total_sequences_to_calculate
    structure_values = [pd.NA] * total_sequences_to_calculate if include_structure else None
    processed_count = 0

    # Create progress manager and progress bar
    progress_manager = get_progress_manager(show_progress=show_progress, quiet=quiet)
    mfe_progress = progress_manager.create_bar(
        total=total_sequences_to_calculate, 
        desc="Folding RNA sequences"
    )

    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_processes)
        results_iterator = pool.imap_unordered(
            get_folding_results_mp_helper_indexed, sequences_with_local_indices
        )

        for local_idx, mfe, struct in results_iterator:
            mfe_values[local_idx] = mfe
            if include_structure and structure_values is not None:
                structure_values[local_idx] = struct

            processed_count += 1
            mfe_progress.update(1)

    except Exception as e:
        logging.error(f"Error during multiprocessing MFE calculation: {e}", exc_info=True)
        df_out = df_to_process.copy()
        df_out["mfe"] = pd.NA
        if include_structure:
            df_out["structure"] = pd.NA
        if pool:
            pool.close()
            pool.join()
        if total_sequences_to_calculate > 0:
            logging.info("")
        return df_out

    finally:
        if pool:
            pool.close()
            pool.join()
        mfe_progress.close()

    logging.info("MFE calculations complete.")

    df_out = df_to_process.copy()
    df_out["mfe"] = mfe_values
    if include_structure and structure_values is not None:
        df_out["structure"] = structure_values

    df_out["mfe"] = pd.to_numeric(df_out["mfe"], errors="coerce")
    return df_out


def load_checkpoint_batch(
    filename: str, include_structure: bool, sequence_col: str
) -> Union[pd.DataFrame, None]:
    """
    Attempts to load and validate a batch file as a checkpoint.
    """
    try:
        _, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lower()
        df_checkpoint = None
        csv_engine = "pyarrow" if pyarrow_available else None

        logging.debug(f"Attempting to load checkpoint file: {filename}")
        if file_ext == ".csv":
            df_checkpoint = pd.read_csv(filename, sep=",", engine=csv_engine)
        elif file_ext == ".tsv":
            df_checkpoint = pd.read_csv(filename, sep="\t", engine=csv_engine)
        elif file_ext == ".parquet":
            if not pyarrow_available:
                logging.warning(
                    f"Cannot load Parquet checkpoint '{filename}', pyarrow not available."
                )
                return None
            df_checkpoint = pd.read_parquet(filename, engine="pyarrow")
        else:
            logging.warning(f"Unknown extension for checkpoint file '{filename}'. Cannot load.")
            return None

        if (
            "mfe" not in df_checkpoint.columns
            or (include_structure and "structure" not in df_checkpoint.columns)
            or sequence_col not in df_checkpoint.columns
        ):
            logging.warning(
                f"Checkpoint file '{filename}' is missing required columns. Will re-process."
            )
            return None

        logging.info(f"Successfully loaded and validated checkpoint: {filename}")
        return df_checkpoint
    except Exception as e:
        logging.warning(
            f"Could not load or validate checkpoint file '{filename}': {e}. Will re-process."
        )
        return None


def write_output_file(df: pd.DataFrame, filename: str):
    """
    Writes a DataFrame to a file, inferring format from extension.
    """
    try:
        _, output_ext = os.path.splitext(filename)
        output_ext = output_ext.lower()

        logging.info(f"Writing output file: {filename} ({len(df)} rows)")

        if output_ext == ".csv":
            df.to_csv(filename, sep=",", index=False)
        elif output_ext == ".tsv":
            df.to_csv(filename, sep="\t", index=False)
        elif output_ext == ".parquet":
            if not pyarrow_available:
                logging.error(
                    f"Cannot write Parquet file '{filename}' because 'pyarrow' is not installed."
                )
                sys.exit(1)
            df.to_parquet(filename, engine="pyarrow", index=False)
        else:
            logging.error(
                f"Unsupported output file format '{output_ext}'. Please use .csv, .tsv, or .parquet."
            )
            sys.exit(1)
        logging.info(f"File saved successfully: {filename}")
    except Exception as e:
        logging.error(f"Error writing output file {filename}: {e}", exc_info=True)
        sys.exit(1)


def calculate_rna_mfe(
    input_data: Union[str, pd.DataFrame],
    sequence_col: str = "Sequence",
    include_structure: bool = False,
    num_processes: Optional[int] = None,
    output_file: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate Minimum Free Energy (MFE) and optionally secondary structure for RNA sequences.

    This function provides a simple API interface to calculate RNA secondary structure
    and MFE using LinearFold. It can process data from files or pandas DataFrames.

    Args:
        input_data: Path to input file (.csv, .tsv, .parquet, .fa, .fasta, .fna, .fas) or pandas DataFrame
        sequence_col: Name of the column containing RNA sequences (ignored for FASTA input)
        include_structure: Whether to include dot-bracket structure notation in output
        num_processes: Number of parallel processes to use. Defaults to CPU count
        output_file: Optional path to save results. Format inferred from extension

    Returns:
        pandas.DataFrame: Results with original data plus 'mfe' column and optionally 'structure' column

    Raises:
        FileNotFoundError: If input file doesn't exist
        KeyError: If sequence column not found
        ValueError: If input parameters are invalid
        RuntimeError: If LinearFold processing fails

    Example:
        >>> from features.mfe_linear import calculate_rna_mfe
        >>> import pandas as pd
        >>>
        >>> # From file
        >>> results = calculate_rna_mfe("sequences.csv", include_structure=True)
        >>>
        >>> # From DataFrame
        >>> df = pd.DataFrame({'Sequence': ['GGUGCUGAUGAU', 'AUUGCUAGC']})
        >>> results = calculate_rna_mfe(df, num_processes=4)
        >>> print(results[['Sequence', 'mfe']].head())
    """
    # Set up logging if not already configured
    setup_logging("INFO")

    # Handle DataFrame input
    if isinstance(input_data, pd.DataFrame):
        if sequence_col not in input_data.columns:
            raise KeyError(f"Column '{sequence_col}' not found in DataFrame")

        # Process DataFrame directly
        result_df = calculate_all_mfe_and_structure(
            input_data,
            sequence_col=sequence_col,
            include_structure=include_structure,
            num_processes=num_processes,
            show_progress=show_progress,
            quiet=quiet,
        )

        if output_file:
            write_output_file(result_df, output_file)

        return result_df

    # Handle file input
    elif isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise FileNotFoundError(f"Input file not found: {input_data}")

        # Use the main processing function with temporary output if none specified
        if output_file is None:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                temp_output = tmp.name

            try:
                result_df = process_mfe_calculations(
                    input_file=input_data,
                    output_file=temp_output,
                    sequence_col=sequence_col,
                    include_structure=include_structure,
                    batch_size=0,  # No batching for API usage
                    num_processes=num_processes,
                )
                return result_df
            finally:
                # Clean up temporary file
                if os.path.exists(temp_output):
                    os.unlink(temp_output)
        else:
            return process_mfe_calculations(
                input_file=input_data,
                output_file=output_file,
                sequence_col=sequence_col,
                include_structure=include_structure,
                batch_size=0,  # No batching for API usage
                num_processes=num_processes,
            )
    else:
        raise ValueError("input_data must be a file path string or pandas DataFrame")


def process_mfe_calculations(
    input_file: str,
    output_file: str,
    sequence_col: str = "Sequence",
    include_structure: bool = False,
    batch_size: int = 0,
    num_processes: Union[int, None] = None,
    show_progress: bool = True,
    quiet: bool = False,
):
    """
    Core logic for calculating MFE and optionally structure for sequences from an input file.
    Supports CSV, TSV, Parquet, and FASTA formats.
    """
    if batch_size < 0:
        logging.error("batch_size cannot be negative.")
        sys.exit(1)

    try:
        _, input_ext = os.path.splitext(input_file)
        input_ext = input_ext.lower()
        logging.info(f"Reading input file: {input_file}")
        csv_engine = "pyarrow" if pyarrow_available else None

        # Check if input is a FASTA file
        if input_ext in [".fa", ".fasta", ".fna", ".fas"]:
            try:
                from Bio import SeqIO
            except ImportError:
                logging.error(
                    "Biopython is required for FASTA file processing. Install with: pip install biopython"
                )
                sys.exit(1)

            logging.info("Processing FASTA input file")
            sequences = []
            seq_ids = []

            try:
                with open(input_file, "r") as handle:
                    for record in SeqIO.parse(handle, "fasta"):
                        seq_ids.append(record.id)
                        sequences.append(str(record.seq))

                if not sequences:
                    logging.error(f"No valid sequences found in FASTA file: {input_file}")
                    sys.exit(1)

                logging.info(f"Successfully read {len(sequences)} sequences from FASTA file")
                full_input_df = pd.DataFrame({"id": seq_ids, sequence_col: sequences})

            except Exception as e:
                logging.error(f"Error parsing FASTA file {input_file}: {e}", exc_info=True)
                sys.exit(1)
        elif input_ext == ".csv":
            full_input_df = pd.read_csv(input_file, sep=",", engine=csv_engine)
        elif input_ext == ".tsv":
            full_input_df = pd.read_csv(input_file, sep="\t", engine=csv_engine)
        elif input_ext == ".parquet":
            if not pyarrow_available:
                logging.error(f"Cannot read Parquet file '{input_file}', 'pyarrow' not installed.")
                sys.exit(1)
            full_input_df = pd.read_parquet(input_file, engine="pyarrow")
        else:
            logging.error(
                f"Unsupported input file format '{input_ext}'. Supported formats: .csv, .tsv, .parquet, .fa, .fasta, .fna, .fas"
            )
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
        empty_df_cols = list(full_input_df.columns) + ["mfe"]
        if include_structure:
            empty_df_cols.append("structure")
        empty_df = pd.DataFrame(columns=empty_df_cols)
        write_output_file(empty_df, output_file)
        return empty_df

    if sequence_col not in full_input_df.columns:
        logging.error(f"Sequence column '{sequence_col}' not found in input file.")
        raise KeyError(f"Sequence column '{sequence_col}' not found.")

    final_results_df = full_input_df.copy()
    final_results_df["mfe"] = pd.NA
    if include_structure:
        final_results_df["structure"] = pd.NA

    rows_needing_calculation_indices = set(range(total_rows))

    if batch_size > 0:
        logging.info("--- Checkpoint Scan (Batch Mode) ---")
        num_expected_batches = math.ceil(total_rows / batch_size)
        output_base, output_ext_for_file = os.path.splitext(output_file)

        processed_rows_from_checkpoints = 0
        for i in range(num_expected_batches):
            batch_start_orig_idx = i * batch_size
            batch_end_orig_idx = min((i + 1) * batch_size, total_rows)
            batch_filename = f"{output_base}_{batch_start_orig_idx + 1}-{batch_end_orig_idx}{output_ext_for_file}"

            if os.path.exists(batch_filename):
                df_checkpoint = load_checkpoint_batch(
                    batch_filename, include_structure, sequence_col
                )
                if df_checkpoint is not None and len(df_checkpoint) == (
                    batch_end_orig_idx - batch_start_orig_idx
                ):
                    logging.info(
                        f"Valid checkpoint loaded for rows {batch_start_orig_idx + 1}-{batch_end_orig_idx}."
                    )
                    indices_to_update = range(batch_start_orig_idx, batch_end_orig_idx)
                    final_results_df.loc[indices_to_update, "mfe"] = df_checkpoint["mfe"].values
                    if include_structure:
                        final_results_df.loc[indices_to_update, "structure"] = df_checkpoint[
                            "structure"
                        ].values
                    rows_needing_calculation_indices.difference_update(indices_to_update)
                    processed_rows_from_checkpoints += len(df_checkpoint)

        if processed_rows_from_checkpoints > 0:
            logging.info(f"{processed_rows_from_checkpoints} rows' data loaded from checkpoints.")
        logging.info(f"{len(rows_needing_calculation_indices)} rows will be processed.")
        logging.info("--- End Checkpoint Scan ---")

    if rows_needing_calculation_indices:
        rows_to_calc_list = sorted(list(rows_needing_calculation_indices))
        df_to_calculate = full_input_df.iloc[rows_to_calc_list].copy()

        # The index of df_to_calculate corresponds to the original row numbers
        calculated_data_df = calculate_all_mfe_and_structure(
            df_to_calculate,
            sequence_col=sequence_col,
            include_structure=include_structure,
            num_processes=num_processes,
            show_progress=show_progress,
            quiet=quiet,
        )
        # Update final_results_df using the original index from calculated_data_df
        final_results_df.update(calculated_data_df)

    else:
        logging.info("No new rows to calculate.")

    if batch_size > 0:
        logging.info(f"--- Writing Output in Batches ({batch_size} rows/batch) ---")
        num_output_batches = math.ceil(total_rows / batch_size)
        output_base, output_ext_for_file = os.path.splitext(output_file)

        for i in range(num_output_batches):
            start_row_idx = i * batch_size
            end_row_idx = min((i + 1) * batch_size, total_rows)
            df_chunk_to_write = final_results_df.iloc[start_row_idx:end_row_idx]
            batch_filename = f"{output_base}_{start_row_idx + 1}-{end_row_idx}{output_ext_for_file}"
            write_output_file(df_chunk_to_write, batch_filename)
        logging.info("--- All batches written. ---")
    else:
        logging.info("--- Writing Output to Single File ---")
        write_output_file(final_results_df, output_file)

    logging.info("MFE processing complete.")
    return final_results_df


def main() -> int:
    """
    Command-line interface for MFE calculation script.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Calculate MFE and optionally RNA secondary structure for sequences in a file using LinearFold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file", help="Path to the input file (.csv, .tsv, .parquet, .fa, .fasta, .fna, .fas)."
    )
    parser.add_argument("output_file", help="Path/base name for the output file(s).")
    parser.add_argument(
        "-s", "--sequence_col", default="Sequence", help="Name of the column with RNA sequences."
    )
    parser.add_argument(
        "--include_structure", action="store_true", help="Include dot-bracket structure in output."
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=0,
        help="Batch size for processing. 0 means single output. >0 enables checkpointing.",
    )
    parser.add_argument(
        "--num_processes",
        "-p",
        type=int,
        default=None,
        help="Number of parallel processes. Defaults to all CPU cores.",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level.",
    )
    
    # Add progress-related CLI arguments
    update_cli_args(parser)

    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Resolve progress settings from CLI arguments
    progress_settings = resolve_progress_settings(args)

    if pyarrow_available:
        logging.info("'pyarrow' package found. Used for Parquet and can accelerate CSV/TSV I/O.")
    else:
        logging.warning(
            "'pyarrow' not found. Parquet support is unavailable. Install with: pip install pandas pyarrow"
        )

    try:
        process_mfe_calculations(
            input_file=args.input_file,
            output_file=args.output_file,
            sequence_col=args.sequence_col,
            include_structure=args.include_structure,
            batch_size=args.batch_size,
            num_processes=args.num_processes,
            **progress_settings,
        )
        logging.info("Script finished successfully.")
        return 0
    except (FileNotFoundError, KeyError, Exception) as e:
        logging.error(f"A critical error occurred: {e}", exc_info=(args.log_level == "DEBUG"))
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
