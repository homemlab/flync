#!/usr/bin/env python3
"""feature_wrapper
===================

Unified orchestration layer for transcript-level feature generation.

This module coordinates multiple feature extraction backends used in the
lncRNA / transcript classification workflow:

    * ``bwq``  - BigWig / BigBed quantitative regional summarisation
    * ``mfe``  - Minimum free energy (RNA secondary structure) features
    * ``kmer`` - High-dimensional k-mer frequency profiles (optionally TF-IDF + SVD)

It provides both a Python API (the :class:`FeatureWrapper`) and a CLI
(``python feature_wrapper.py <command>``) that expose consistent parameters,
progress reporting, and persistent caching of remote BigWig resources.

Pipeline Execution Order (``run_all``):
    1. GTF Input Processing (required):
        a. Auto-generate BED for BWQ from transcript-level features
        b. Extract properly spliced sequences using gffutils
    2. BigWig Query (regional signal extraction using BED)
    3. RNA MFE / structure prediction
    4. K-mer profiling (dense or sparse) followed by optional transformations:
        a. Dimensionality reduction (Truncated SVD) - either global or per k length
        b. TF-IDF weighting (applied pre- or post-reduction depending on configuration)
    5. Final keyed merge on ``transcript_id``

GTF-Based Workflow (Required):
    The pipeline requires GTF files as the primary input for accurate sequence extraction.
    GTF-based extraction:
    * Properly splices multi-exon transcripts (exon concatenation)
    * Handles strand orientation automatically (reverse complement for minus strand)
    * Converts DNA to RNA (T → U)
    * Caches gffutils database for faster reruns
    
    When GTF is provided:
    * BED file is auto-generated for BWQ (transcript-level coordinates only)
    * Sequences are extracted using gffutils for proper splicing
    * BED and FASTA can be provided explicitly to override auto-generation

K-mer Transformation Strategy:
    The heavy k-mer space can be processed in two principal modes:
    * Global SVD: one decomposition over the concatenated feature space.
    * Grouped SVD: separate SVD per k (e.g. all 3-mers, all 4-mers) to balance
      representation when long k-mers dominate raw dimensionality.
    Optional TF-IDF weighting modulates raw counts prior to reduction (or alone
    if reduction disabled). Transformed sparse matrices are stored in
    ``DataFrame.attrs`` instead of column-materialising when ``sparse=True``.

Caching:
    Remote BigWig / BigBed assets are downloaded into a persistent directory
    (``bwq_tracks/bwq_persistent_cache`` by default) enabling repeatable and
    faster executions. GTF databases are cached separately in ``gffutils_cache``
    subdirectory. Cache lifecycle is controlled via CLI flags or the API.

Progress Reporting:
    Multi-phase progress bars can be enabled/disabled; a quiet mode suppresses
    bars and most log lines for batch / automated contexts.

Error Handling Philosophy:
    * Input validation raises ``FileNotFoundError`` / ``ValueError`` early.
    * Backend failures are wrapped as ``RuntimeError`` with contextual messages.
    * Non-critical merge alignment issues log warnings (e.g. missing key columns).

Example (CLI):
    GTF-based workflow (recommended)::

        python feature_wrapper.py all \
            --gtf annotations.gtf \
            --ref-genome genome.fa \
            --bwq-config config.yaml \
            --use-dim-redux --redux-n-components 32 --use-tfidf --sparse \
            -o features.parquet

    Legacy BED-based workflow::

        python feature_wrapper.py all \
            --bed regions.bed \
            --ref-genome genome.fa \
            --bwq-config config.yaml \
            -o features.parquet

    Basic BigWig extraction::

        python feature_wrapper.py bwq --bed regions.bed --bwq-config config.yaml -o bwq.parquet

Example (Python API)::

    from features.feature_wrapper import FeatureWrapper
    fw = FeatureWrapper(cache_dir="./bwq_tracks", keep_downloaded_files=True, threads=8)
    
    # GTF-based workflow
    df_all = fw.run_all(
        gtf_file="annotations.gtf",
        ref_genome_path="genome.fa",
        config_file="config.yaml",
        k_min=3, k_max=12,
        use_dim_redux=True, redux_n_components=32, use_tfidf=True,
        sparse=True, group_kmer_redux_by_length=True
    )
    
    # Or BED-based workflow
    df_all = fw.run_all(
        bed_file="regions.bed",
        ref_genome_path="genome.fa",
        config_file="config.yaml",
        k_min=3, k_max=12,
        use_dim_redux=True, redux_n_components=32, use_tfidf=True,
        sparse=True, group_kmer_redux_by_length=True
    )

Design Notes:
    * The module intentionally avoids persisting intermediate artefacts (except
      cache + optional k-mer sparse matrices embedded in ``attrs``) to reduce IO.
    * All public methods return pandas DataFrames unless explicitly documented.
    * The code refrains from modifying original input DataFrames in place except
      where renaming is needed before merging; callers should treat returned
      DataFrames as new objects.
"""

import argparse
import logging
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import pyranges as pr
from scipy import sparse

# Import progress utilities
try:
    from src.utils.progress import (
        get_progress_manager,
        update_cli_args,
        resolve_progress_settings,
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
    )

# Import feature extraction modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import bwq, kmer, mfe

try:
    from utils.kmer_redux import apply_kmer_transformations
except ImportError:
    apply_kmer_transformations = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("feature_wrapper")


class FeatureWrapper:
    """High-level orchestrator for feature module execution and aggregation.

    Responsibilities:
    * Standardise execution of individual feature providers (bwq, mfe, kmer).
    * Provide a coherent logging / progress UX across heterogeneous backends.
    * Manage on-disk caching of remote signal tracks (BigWig / BigBed).
    * Support GTF-based workflow for accurate spliced transcript sequence extraction (required).
    * Apply optional k-mer post-processing (TF-IDF, SVD - grouped or global).
    * Perform schema harmonisation (column renames) and keyed merges.

    GTF-Based Features (Required):
    * Extract properly spliced transcript sequences using gffutils
    * Auto-generate BED files from GTF for BWQ processing
    * Cache gffutils databases for fast reruns
    * GTF input is required for the run_all() pipeline

    Attribute Summary:
        base_cache_dir (str): Root directory for any persisted track resources and gffutils databases.
        keep_downloaded_files (bool): If False, remote assets are redownloaded every call.
        clear_cache_on_startup (bool): If True, existing cache is purged at init.
        show_progress (bool): Enables progress bar display.
        quiet (bool): Suppresses most non-error logs and progress indicators.
        default_threads (int | None): Shared worker budget applied to downstream modules.

    Thread / Process Safety:
        The wrapper itself is stateless aside from cache directory configuration;
        concurrent instances may reuse the cache, but no file locking is enforced.
        External synchronisation is advised for highly parallel cluster usage.
    """

    def __init__(
        self,
        log_level: str = "INFO",
        cache_dir: Optional[str] = None,
        keep_downloaded_files: bool = True,
        clear_cache_on_startup: bool = False,
        show_progress: bool = True,
        quiet: bool = False,
        threads: Optional[int] = None,
    ):
        """Create a wrapper instance.

        Parameters
        ----------
        log_level : str, default "INFO"
            Logging verbosity (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``).
        cache_dir : str | None
            Base directory for BigWig cache. Defaults to ``./bwq_tracks`` if ``None``.
        keep_downloaded_files : bool, default True
            Retain remote downloads for reuse across invocations.
        clear_cache_on_startup : bool, default False
            Purge existing cache directory during construction.
        show_progress : bool, default True
            Display progress bars (ignored if ``quiet=True``).
        quiet : bool, default False
            Suppress progress bars and most informational logging.
        threads : int | None, optional
            Default worker count to reuse across feature modules. When ``None`` each
            backend falls back to its internal CPU-based heuristic.
        """
        self.set_log_level(log_level)

        # Set up caching configuration
        if cache_dir is None:
            # Use a default base directory in the workspace, BWQ will create subdirectories
            self.base_cache_dir = os.path.join(os.getcwd(), "bwq_tracks")
        else:
            # Use the specified directory as the base directory
            self.base_cache_dir = cache_dir

        self.keep_downloaded_files = keep_downloaded_files
        self.clear_cache_on_startup = clear_cache_on_startup

        # Store progress settings
        self.show_progress = show_progress
        self.quiet = quiet

        if threads is not None and threads < 1:
            raise ValueError("threads must be a positive integer when provided.")
        self.default_threads = threads

        # Create base cache directory if it doesn't exist
        if self.keep_downloaded_files:
            os.makedirs(self.base_cache_dir, exist_ok=True)
            logger.info(f"Using base cache directory: {self.base_cache_dir}")

        logger.info("Initialized FeatureWrapper")

    def set_log_level(self, log_level: str) -> None:
        """Adjust logging level for this wrapper and root handlers.

        Parameters
        ----------
        log_level : str
            Desired logging level name. Falls back to ``INFO`` if invalid.
        """
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            logger.warning(f"Invalid log level '{log_level}'. Defaulting to INFO.")
            numeric_level = logging.INFO

        # Set log level for this module
        logger.setLevel(numeric_level)

        # Set log level for all handlers of the root logger
        for handler in logging.getLogger().handlers:
            handler.setLevel(numeric_level)

    def clear_cache(self) -> None:
        """Remove persisted BigWig / BigBed cached files (if caching enabled)."""
        if self.keep_downloaded_files:
            # The actual cache is in base_cache_dir/bwq_persistent_cache
            actual_cache_dir = os.path.join(self.base_cache_dir, "bwq_persistent_cache")
            if os.path.exists(actual_cache_dir):
                import shutil

                logger.info(f"Clearing cache directory: {actual_cache_dir}")
                shutil.rmtree(actual_cache_dir)
                os.makedirs(actual_cache_dir, exist_ok=True)
            else:
                logger.info("No cache directory to clear")
        else:
            logger.info("Caching is disabled")

    def get_cache_info(self) -> Dict[str, Union[str, int, bool]]:
        """Summarise cache configuration and current contents.

        Returns
        -------
        dict
            Includes flags, directory paths, file count, size statistics (MB),
            and a short file name listing when accessible.
        """
        # The actual cache is in base_cache_dir/bwq_persistent_cache
        actual_cache_dir = os.path.join(self.base_cache_dir, "bwq_persistent_cache")

        info = {
            "cache_enabled": self.keep_downloaded_files,
            "base_cache_directory": self.base_cache_dir,
            "cache_directory": actual_cache_dir,
            "cache_exists": os.path.exists(actual_cache_dir)
            if self.keep_downloaded_files
            else False,
        }

        if self.keep_downloaded_files and os.path.exists(actual_cache_dir):
            try:
                cache_files = [
                    f
                    for f in os.listdir(actual_cache_dir)
                    if os.path.isfile(os.path.join(actual_cache_dir, f))
                ]
                info["cached_files_count"] = len(cache_files)
                info["cached_files"] = cache_files

                # Calculate total cache size
                total_size = 0
                for filename in cache_files:
                    filepath = os.path.join(actual_cache_dir, filename)
                    total_size += os.path.getsize(filepath)
                info["cache_size_bytes"] = total_size
                info["cache_size_mb"] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not read cache directory: {e}")
                info["error"] = str(e)

        return info

    def run_bwq(
        self,
        bed_file: str,
        bigwig_files: Optional[List[str]] = None,
        config_file: Optional[str] = None,
        output_file: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Execute BigWig / BigBed quantitative feature extraction.

        Parameters
        ----------
        bed_file : str
            BED6 / BED12 file with genomic intervals to summarise.
        bigwig_files : list[str] | None, optional
            Explicit list of BigWig / BigBed URLs or paths. Ignored when
            ``config_file`` is supplied.
        config_file : str | None, optional
            YAML specification describing remote/source BigWig resources and
            requested summary statistics. Takes precedence over ``bigwig_files``.
        output_file : str | None
            Optional parquet destination. Not written if omitted.
        threads : int | None, optional
            Worker count override for BigWig querying. Falls back to the wrapper
            default thread value when omitted.
        **kwargs : any
            Forwarded to internal ``bwq.process_bigwig_query`` / ``process_bed_regions``.

        Returns
        -------
        pandas.DataFrame
            Table keyed by region transcript identifier containing signal features.

        Raises
        ------
        FileNotFoundError
            If required input files (BED / config / explicit BigWigs) are missing.
        RuntimeError
            On downstream processing errors.
        """
        logger.info("Running BigWig Query feature extraction...")

        # Validate input files
        if not os.path.exists(bed_file):
            raise FileNotFoundError(f"BED file not found: {bed_file}")

        if config_file:
            if not os.path.exists(config_file) and not config_file.startswith(
                ("http://", "https://", "ftp://")
            ):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            logger.info(f"Using configuration file: {config_file}")
        elif bigwig_files:
            for bw_file in bigwig_files:
                if not os.path.exists(bw_file) and not bw_file.startswith(
                    ("http://", "https://", "ftp://")
                ):
                    raise FileNotFoundError(f"BigWig file not found: {bw_file}")
        else:
            raise ValueError("Either bigwig_files or config_file must be provided")

        try:
            threads_override = kwargs.pop("threads", None)
            effective_threads = (
                threads
                if threads is not None
                else threads_override
                if threads_override is not None
                else self.default_threads
            )
            if effective_threads is not None and effective_threads < 1:
                raise ValueError("threads must be a positive integer when provided.")

            if config_file:
                # Use the process_bigwig_query function with config file and caching settings
                result_df = bwq.process_bigwig_query(
                    bed_file=bed_file,
                    config_file=config_file,
                    output_file=None,  # We'll handle saving separately
                    keep_downloaded_files=self.keep_downloaded_files,
                    base_storage_directory=self.base_cache_dir
                    if self.keep_downloaded_files
                    else None,
                    clear_cache_on_startup=self.clear_cache_on_startup,
                    show_progress=self.show_progress,
                    quiet=self.quiet,
                    threads=effective_threads,
                    **kwargs,
                )
            else:
                # Create a BigWigFileManager instance with caching settings
                file_manager = bwq.BigWigFileManager(
                    keep_downloaded_files=self.keep_downloaded_files,
                    base_storage_directory=self.base_cache_dir
                    if self.keep_downloaded_files
                    else None,
                )

                # Process the query using functions from bwq module
                result_df = bwq.process_bed_regions(
                    bed_file=bed_file,
                    bigwig_files=bigwig_files,
                    file_manager=file_manager,
                    show_progress=self.show_progress,
                    quiet=self.quiet,
                    **kwargs,
                )

            # Save output if requested
            if output_file:
                logger.info(f"Saving BigWig Query results to {output_file}")
                result_df.to_parquet(output_file)

            return result_df

        except Exception as e:
            logger.error(f"BigWig Query extraction failed: {e}")
            raise RuntimeError(f"BigWig Query extraction failed: {e}") from e

    def run_mfe(
        self,
        df_input: pd.DataFrame,
        sequence_col: str = "Sequence",
        include_structure: bool = False,
        num_processes: Optional[int] = None,
        output_file: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Compute RNA minimum free energy (and optional structure) features.

        Parameters
        ----------
        df_input : pandas.DataFrame
            Must contain a sequence column of RNA (U instead of T) strings.
        sequence_col : str, default "Sequence"
            Name of column holding input sequences.
        include_structure : bool, default False
            If True, predicted dot-bracket secondary structure is retained.
        num_processes : int | None
            Worker process count. Defaults to ``os.cpu_count()`` inside backend.
        output_file : str | None
            Optional parquet sink.
        **kwargs : any
            Forwarded to ``mfe.calculate_all_mfe_and_structure``.

        Returns
        -------
        pandas.DataFrame
            Contains ``mfe`` (later renamed ``ss_mfe``) and optionally ``structure``.

        Raises
        ------
        KeyError
            If ``sequence_col`` is absent.
        RuntimeError
            On backend failure.
        """
        logger.info("Running MFE feature extraction...")

        # Validate input DataFrame
        if sequence_col not in df_input.columns:
            raise KeyError(
                f"Sequence column '{sequence_col}' not found in the input DataFrame."
            )

        try:
            threads_override = kwargs.pop("threads", None)
            effective_processes = (
                num_processes
                if num_processes is not None
                else threads_override
                if threads_override is not None
                else self.default_threads
            )
            if effective_processes is not None and effective_processes < 1:
                raise ValueError(
                    "num_processes/threads must be a positive integer when provided."
                )

            # Run MFE calculation using functions from mfe module
            result_df = mfe.calculate_all_mfe_and_structure(
                df_to_process=df_input,
                sequence_col=sequence_col,
                include_structure=include_structure,
                num_processes=effective_processes,
                show_progress=self.show_progress,
                quiet=self.quiet,
                **kwargs,
            )

            # Save output if requested
            if output_file:
                logger.info(f"Saving MFE results to {output_file}")
                result_df.to_parquet(output_file)

            return result_df

        except Exception as e:
            logger.error(f"MFE calculation failed: {e}")
            raise RuntimeError(f"MFE calculation failed: {e}") from e

    def run_kmer(
        self,
        input_path: str,
        k_min: int = 3,
        k_max: int = 12,
        output_format: str = "dataframe",
        output_file: Optional[str] = None,
        return_sparse_paths: bool = False,
        sparse_base_name: Optional[str] = None,
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, str]]]:
        """Generate raw k-mer count features with optional raw sparse artefact persistence.

        Parameters
        ----------
        input_path : str
            FASTA file (or directory of FASTA files) containing sequences.
        k_min : int, default 3
            Smallest k considered.
        k_max : int, default 12
            Largest k considered.
        output_format : {"dataframe","sparse_dataframe","matrix"}, default "dataframe"
            Desired return representation. ``matrix`` yields sparse matrix which
            is then converted here to a sparse DataFrame.
        output_file : str | None
            Optional parquet output path.
        return_sparse_paths : bool, default False
            If True, persist the raw sparse matrix (CSR) plus separate row and
            column name listings to disk and return a tuple ``(df, paths)``.
            This enables deferred TF-IDF / SVD transformation in a later run
            without re-counting k-mers.
        sparse_base_name : str | None
            Base path (no suffix) for saved artefacts. Files written:
            ``<base>_sparse.npz``, ``<base>_rows.txt``, ``<base>_cols.txt``.
            If omitted, a deterministic name under ``<cache>/kmer_temp`` is used.
        num_workers : int | None, optional
            Worker process count override for k-mer counting. Defaults to the
            wrapper-level threads setting when not provided.
        **kwargs : any
            Extra arguments forwarded to ``kmer.calculate_kmer_profiles``.

        Returns
        -------
        pandas.DataFrame or (DataFrame, dict)
            DataFrame of k-mer counts keyed by sequence ID. If
            ``return_sparse_paths`` is True a tuple is returned where the dict
            contains keys: ``sparse_matrix``, ``rows``, ``cols``.

        Raises
        ------
        FileNotFoundError
            If ``input_path`` is absent.
        ValueError
            If ``k_min`` > ``k_max``.
        RuntimeError
            On k-mer computation failure.
        """
        logger.info("Running K-mer feature extraction...")

        # Validate input path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Validate k-mer parameters
        if k_min > k_max:
            raise ValueError(
                f"k_min ({k_min}) must be less than or equal to k_max ({k_max})"
            )

        try:
            # Map feature_wrapper output format to kmer module return_type
            format_mapping = {
                "dataframe": "dense_df",
                "sparse_dataframe": "sparse_df",
                "matrix": "sparse_matrix",
            }
            kmer_return_type = format_mapping.get(output_format, "sparse_df")

            # Use cache directory for k-mer temporary files instead of system /tmp
            # This ensures the temp directory persists and is under user control
            kmer_temp_dir = os.path.join(self.base_cache_dir, "kmer_temp")
            os.makedirs(kmer_temp_dir, exist_ok=True)

            # Run k-mer calculation using functions from kmer module
            threads_override = kwargs.pop("threads", None)
            effective_workers = (
                num_workers
                if num_workers is not None
                else threads_override
                if threads_override is not None
                else self.default_threads
            )
            if effective_workers is not None and effective_workers < 1:
                raise ValueError(
                    "num_workers/threads must be a positive integer when provided."
                )

            result = kmer.calculate_kmer_profiles(
                input_path=input_path,
                k_min=k_min,
                k_max=k_max,
                num_workers=effective_workers,
                return_type=kmer_return_type,
                temp_dir=kmer_temp_dir,
                show_progress=self.show_progress,
                quiet=self.quiet,
                **kwargs,
            )

            # Convert to DataFrame if result is a tuple (sparse_matrix format)
            sparse_paths: Dict[str, str] = {}
            if output_format == "matrix":
                sparse_matrix, sequence_ids, kmer_names = result  # type: ignore
                result_df = pd.DataFrame.sparse.from_spmatrix(
                    sparse_matrix, index=sequence_ids, columns=kmer_names
                )
                if return_sparse_paths:
                    base = sparse_base_name or os.path.join(
                        kmer_temp_dir, f"kmer_k{k_min}_{k_max}"
                    )
                    from scipy.sparse import save_npz

                    save_npz(f"{base}_sparse.npz", sparse_matrix)
                    with open(f"{base}_rows.txt", "w") as f_rows:
                        f_rows.write("\n".join(sequence_ids))
                    with open(f"{base}_cols.txt", "w") as f_cols:
                        f_cols.write("\n".join(kmer_names))
                    sparse_paths = {
                        "sparse_matrix": f"{base}_sparse.npz",
                        "rows": f"{base}_rows.txt",
                        "cols": f"{base}_cols.txt",
                    }
            else:
                result_df = result  # type: ignore
                # If user wants raw sparse paths but we only have a DataFrame,
                # convert to CSR and persist.
                if return_sparse_paths and output_format == "sparse_dataframe":
                    from scipy.sparse import csr_matrix, save_npz

                    base = sparse_base_name or os.path.join(
                        kmer_temp_dir, f"kmer_k{k_min}_{k_max}"
                    )
                    csr = csr_matrix(result_df.sparse.to_coo())  # type: ignore
                    save_npz(f"{base}_sparse.npz", csr)
                    with open(f"{base}_rows.txt", "w") as f_rows:
                        f_rows.write("\n".join(result_df.index.astype(str)))
                    with open(f"{base}_cols.txt", "w") as f_cols:
                        f_cols.write("\n".join(result_df.columns.astype(str)))
                    sparse_paths = {
                        "sparse_matrix": f"{base}_sparse.npz",
                        "rows": f"{base}_rows.txt",
                        "cols": f"{base}_cols.txt",
                    }

            # Save output if requested
            if output_file:
                logger.info(f"Saving K-mer results to {output_file}")
                result_df.to_parquet(output_file)

            if return_sparse_paths:
                return result_df, sparse_paths
            return result_df

        except Exception as e:
            logger.error(f"K-mer calculation failed: {e}")
            raise RuntimeError(f"K-mer calculation failed: {e}") from e

    def _get_gffutils_cache_path(self, gtf_file: str) -> str:
        """Generate cache database path for a GTF file.

        Uses MD5 hash of GTF file contents for cache key to detect file changes.

        Parameters
        ----------
        gtf_file : str
            Path to GTF file

        Returns
        -------
        str
            Full path to cache database file
        """
        import hashlib

        # Hash file contents, not path, to detect changes
        abs_path = os.path.abspath(gtf_file)
        md5_hash = hashlib.md5()

        with open(abs_path, "rb") as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)

        gtf_hash = md5_hash.hexdigest()
        cache_dir = os.path.join(self.base_cache_dir, "gffutils_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{gtf_hash}.db")

    def gtf_transcripts_to_bed(
        self,
        gtf_file: str,
        output_bed: str,
        feature_type: str = "transcript",
    ) -> str:
        """Convert GTF transcript features to BED format for BWQ processing.

        Creates BED6 format: chrom, start, end, transcript_id, score, strand.
        Only includes transcript-level features (not exons).

        Parameters
        ----------
        gtf_file : str
            Path to input GTF file
        output_bed : str
            Path for output BED file
        feature_type : str, default "transcript"
            GTF feature type to extract

        Returns
        -------
        str
            Path to created BED file

        Raises
        ------
        FileNotFoundError
            If GTF file doesn't exist
        ValueError
            If no transcripts found in GTF
        """
        logger.info(f"Converting GTF to BED: {gtf_file} -> {output_bed}")

        # Validate input
        if not os.path.exists(gtf_file):
            raise FileNotFoundError(f"GTF file not found: {gtf_file}")

        try:
            import pyranges as pr
        except ImportError:
            logger.error(
                "pyranges is required for GTF to BED conversion. "
                "Install with: pip install pyranges"
            )
            raise ImportError("pyranges is required")

        # Read GTF file
        try:
            gtf_data = pr.read_gtf(gtf_file)
            logger.info(f"Loaded GTF with {len(gtf_data)} total features")
        except Exception as e:
            logger.error(f"Failed to read GTF file: {e}")
            raise ValueError(f"Invalid GTF file format: {e}")

        # Filter for transcript-level features
        gtf_df = gtf_data.df
        if "Feature" not in gtf_df.columns:
            raise ValueError("GTF file missing 'Feature' column")

        transcript_df = gtf_df[gtf_df["Feature"] == feature_type].copy()

        if len(transcript_df) == 0:
            raise ValueError(f"No {feature_type} features found in GTF file")

        logger.info(f"Found {len(transcript_df)} {feature_type} features")

        # Extract required columns for BED6 format
        if "transcript_id" not in transcript_df.columns:
            raise ValueError("GTF file missing 'transcript_id' attribute")

        # Create BED6 DataFrame
        bed_df = pd.DataFrame(
            {
                "chrom": transcript_df["Chromosome"],
                "start": transcript_df["Start"],
                "end": transcript_df["End"],
                "name": transcript_df["transcript_id"],
                "score": ".",
                "strand": transcript_df["Strand"],
            }
        )

        # Write to BED file
        bed_df.to_csv(output_bed, sep="\t", header=False, index=False)
        logger.info(f"Wrote {len(bed_df)} transcripts to BED file: {output_bed}")

        return output_bed

    def extract_transcripts_from_gtf(
        self,
        gtf_file: str,
        ref_genome_path: str,
        output_fasta: str,
        return_df: bool = False,
        id_column: str = "transcript_id",
        feature_type: str = "transcript",
        use_cache: bool = True,
        **kwargs,
    ) -> Union[str, Tuple[str, pd.DataFrame]]:
        """Extract spliced transcript sequences from GTF using gffutils.

        This method:
        - Creates/loads cached gffutils database
        - Extracts properly spliced transcript sequences
        - Handles strand orientation automatically
        - Converts DNA to RNA (T → U)
        - Warns about incomplete transcripts

        Parameters
        ----------
        gtf_file : str
            Path to GTF annotation file
        ref_genome_path : str
            Path to reference genome FASTA
        output_fasta : str
            Destination FASTA path for transcript sequences
        return_df : bool, default False
            If True, return DataFrame with sequences
        id_column : str, default "transcript_id"
            Column name for transcript identifiers
        feature_type : str, default "transcript"
            GTF feature type to extract
        use_cache : bool, default True
            Whether to cache gffutils database
        **kwargs : any
            Reserved for future expansion

        Returns
        -------
        str or (str, DataFrame)
            FASTA path, or tuple if return_df=True

        Raises
        ------
        ImportError
            If gffutils/pyfaidx not installed
        FileNotFoundError
            If GTF or genome FASTA missing
        RuntimeError
            If sequence extraction fails
        """
        logger.info(f"Extracting spliced sequences from GTF: {gtf_file}")

        # Check for required dependencies
        try:
            import gffutils
        except ImportError:
            logger.error(
                "gffutils is required for GTF-based sequence extraction. "
                "Install with: pip install gffutils"
            )
            raise ImportError("gffutils is required")

        try:
            import pyfaidx
        except ImportError:
            logger.error(
                "pyfaidx is required for sequence extraction. "
                "Install with: pip install pyfaidx"
            )
            raise ImportError("pyfaidx is required")

        # Validate input files
        if not os.path.exists(gtf_file):
            raise FileNotFoundError(f"GTF file not found: {gtf_file}")

        if not os.path.exists(ref_genome_path):
            raise FileNotFoundError(f"Reference genome not found: {ref_genome_path}")

        # Get or create gffutils database
        db_path = self._get_gffutils_cache_path(gtf_file)

        if use_cache and os.path.exists(db_path):
            logger.info(f"Loading cached gffutils database: {db_path}")
            try:
                db = gffutils.FeatureDB(db_path)
            except Exception as e:
                logger.warning(f"Failed to load cached database: {e}. Recreating...")
                os.remove(db_path)
                db = None
        else:
            db = None

        if db is None:
            logger.info("Creating gffutils database from GTF...")
            try:
                db = gffutils.create_db(
                    gtf_file,
                    dbfn=db_path if use_cache else ":memory:",
                    force=True,
                    keep_order=True,
                    merge_strategy="merge",
                    sort_attribute_values=True,
                    disable_infer_transcripts=True,  # Don't infer transcripts from exons
                    disable_infer_genes=True,  # Don't infer genes from transcripts
                )
                logger.info("gffutils database created successfully")
            except Exception as e:
                logger.error(f"Failed to create gffutils database: {e}")
                raise RuntimeError(f"gffutils database creation failed: {e}")

        # Extract sequences
        total_transcripts = 0
        empty_transcripts = 0
        failed_transcripts = 0
        sequences_written = 0

        logger.info(f"Extracting sequences for {feature_type} features...")

        try:
            with open(output_fasta, "w") as fasta_handle:
                for transcript in db.features_of_type(feature_type):
                    total_transcripts += 1

                    try:
                        # Get transcript ID
                        if "transcript_id" in transcript.attributes:
                            transcript_id = transcript.attributes["transcript_id"][0]
                        else:
                            transcript_id = transcript.id

                        # Check if transcript has exons
                        exons = list(db.children(transcript, featuretype="exon"))
                        if not exons:
                            logger.warning(
                                f"Transcript {transcript_id} has no exons in GTF. "
                                f"Sequence extraction may fail or return genomic span."
                            )

                        # Extract sequence
                        sequence = transcript.sequence(ref_genome_path, use_strand=True)

                        if not sequence or len(sequence) == 0:
                            empty_transcripts += 1
                            logger.warning(
                                f"Empty sequence extracted for {transcript_id}"
                            )
                            continue

                        # Convert DNA to RNA (T -> U)
                        rna_sequence = sequence.upper().replace("T", "U")

                        # Write to FASTA
                        fasta_handle.write(f">{transcript_id}\n")
                        # Write in 60-character lines
                        for i in range(0, len(rna_sequence), 60):
                            fasta_handle.write(f"{rna_sequence[i : i + 60]}\n")

                        sequences_written += 1

                    except Exception as e:
                        failed_transcripts += 1
                        logger.warning(f"Failed to extract {transcript.id}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error writing FASTA file: {e}")
            raise RuntimeError(f"Failed to write sequences: {e}")

        # Summary logging
        logger.info(
            f"Processed {total_transcripts} transcripts: "
            f"{sequences_written} written, {empty_transcripts} empty, "
            f"{failed_transcripts} failed"
        )

        if sequences_written == 0:
            raise RuntimeError("No sequences were successfully extracted")

        # Optionally return DataFrame
        if return_df:
            sequence_df = self._load_fasta_to_dataframe(
                output_fasta, id_column=id_column
            )
            return output_fasta, sequence_df

        return output_fasta

    @staticmethod
    def _load_fasta_to_dataframe(
        fasta_path: str,
        id_column: str = "transcript_id",
    ) -> pd.DataFrame:
        """Load sequences from a FASTA file into a tidy DataFrame.

        Parameters
        ----------
        fasta_path : str
            Path to the FASTA file to load.
        id_column : str, default "transcript_id"
            Name to assign to the identifier column in the resulting DataFrame.

        Returns
        -------
        pandas.DataFrame
            Two-column DataFrame ``[id_column, 'Sequence']`` with RNA sequences
            (T → U conversion applied).

        Raises
        ------
        FileNotFoundError
            If the FASTA file does not exist.
        RuntimeError
            If no records can be read from the FASTA file.
        """
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        identifiers: List[str] = []
        sequences: List[str] = []
        current_id: Optional[str] = None
        current_seq: List[str] = []

        with open(fasta_path, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if current_id is not None:
                        seq = "".join(current_seq).upper().replace("T", "U")
                        identifiers.append(current_id)
                        sequences.append(seq)
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

        if current_id is not None:
            seq = "".join(current_seq).upper().replace("T", "U")
            identifiers.append(current_id)
            sequences.append(seq)

        if not identifiers:
            raise RuntimeError(f"No FASTA records found in {fasta_path}")

        fasta_df = pd.DataFrame({id_column: identifiers, "Sequence": sequences})
        return fasta_df

    def extract_transcripts_from_bed(
        self,
        bed_file: str,
        output_fasta: str,
        ref_genome_path: str,
        return_df: bool = False,
        name_column: str = "Name",
        id_column: str = "transcript_id",
        **kwargs,
    ) -> Union[str, Tuple[str, pd.DataFrame]]:
        """Extract stranded transcript sequences from genomic intervals.

        Sequences are retrieved using ``pyranges.get_sequence`` and converted to
        RNA alphabet (T → U). A FASTA is always written; optionally a tidy
        DataFrame of IDs and sequences is also returned.

        Parameters
        ----------
        bed_file : str
            BED file describing transcript intervals.
        output_fasta : str
            Destination FASTA path.
        ref_genome_path : str
            Reference genome FASTA used for sequence lookup.
        return_df : bool, default False
            If True, also return a DataFrame with columns ``[id_column, 'Sequence']``.
        name_column : str, default "Name"
            Column in BED to use as sequence identifier; synthetic IDs generated
            if absent.
        id_column : str, default "transcript_id"
            Name to assign to identifier column in returned DataFrame.
        **kwargs : any
            Reserved for future expansion (currently unused).

        Returns
        -------
        str | (str, pandas.DataFrame)
            FASTA path, or tuple of FASTA path and sequence DataFrame when
            ``return_df`` is True.

        Raises
        ------
        ImportError
            If ``pyranges`` / ``pyfaidx`` are not installed.
        FileNotFoundError
            If BED or reference FASTA is missing.
        RuntimeError
            If no sequences could be extracted or write fails.
        """
        try:
            import pandas as pd
            import pyranges as pr
        except ImportError:
            logger.error(
                "pyranges is required for sequence extraction. Install with: pip install pyranges"
            )
            raise ImportError("pyranges is required for sequence extraction")

        try:
            import pyfaidx
        except ImportError:
            logger.error(
                "pyfaidx is required for sequence extraction. Install with: pip install pyfaidx"
            )
            raise ImportError("pyfaidx is required for sequence extraction")

        logger.info(f"Extracting RNA sequences from BED file: {bed_file}")

        # Check if reference genome exists
        if not os.path.exists(ref_genome_path):
            logger.error(f"Reference genome file not found: {ref_genome_path}")
            raise FileNotFoundError(
                f"Reference genome file not found: {ref_genome_path}"
            )

        # Load BED file using pyranges
        try:
            bed_ranges = pr.read_bed(bed_file)
            logger.info(f"Loaded {len(bed_ranges)} regions from BED file")

            # Ensure the BED file has a Name column
            if name_column not in bed_ranges.columns:
                logger.warning(
                    "BED file does not have a Name column. Using genomic coordinates as names."
                )
                bed_ranges[name_column] = bed_ranges.apply(
                    lambda x: f"{x.Chromosome}:{x.Start}-{x.End}"
                )

        except Exception as e:
            logger.error(f"Error reading BED file: {e}")
            raise

        # Extract sequences using pyranges get_sequence
        try:
            logger.info(
                f"Extracting sequences using reference genome: {ref_genome_path}"
            )
            bed_ranges.Sequence = pr.get_sequence(bed_ranges, ref_genome_path)

            # Convert DNA to RNA (replace T with U)
            bed_ranges.Sequence = bed_ranges.Sequence.str.replace("T", "U")

        except Exception as e:
            logger.error(f"Error extracting sequences: {e}")
            raise RuntimeError(f"Failed to extract sequences: {e}")

        # Write sequences to FASTA file
        if len(bed_ranges) == 0 or not any(bed_ranges.Sequence):
            logger.error("No sequences extracted from BED regions")
            raise RuntimeError("No sequences extracted from BED regions")

        logger.info(f"Writing {len(bed_ranges)} sequences to {output_fasta}")

        try:
            with open(output_fasta, "w") as f:
                for _, row in bed_ranges.df.iterrows():
                    seq_id = row[name_column]
                    sequence = row["Sequence"]
                    if sequence:
                        f.write(f">{seq_id}\n")
                        for i in range(0, len(sequence), 60):
                            f.write(f"{sequence[i : i + 60]}\n")
        except Exception as e:
            logger.error(f"Error writing FASTA file: {e}")
            raise

        logger.info(
            f"Successfully extracted {len(bed_ranges)} sequences to {output_fasta}"
        )

        if return_df:
            # Build a tidy DataFrame with transcript_id and Sequence
            seq_df = bed_ranges.df[[name_column, "Sequence"]].rename(
                columns={name_column: id_column}
            )
            return output_fasta, seq_df
        return output_fasta

    def load_and_transform_kmer_from_paths(
        self,
        base_path: str,
        use_dim_redux: bool = True,
        redux_n_components: Union[int, Dict[int, int]] = 1,
        use_tfidf: bool = True,
        sparse: bool = True,
        group_kmer_redux_by_length: bool = True,
    ) -> Union[pd.DataFrame, sparse.csr_matrix]:
        """Load previously persisted raw k-mer sparse artefacts and apply transformations.

        Parameters
        ----------
        base_path : str
            Base path supplied during persistence (without the ``_sparse.npz``
            suffix). The loader searches for: ``<base>_sparse.npz``,
            ``<base>_rows.txt``, ``<base>_cols.txt`` (and internal variants
            accepted by ``load_kmer_results`` if produced by batch mode).
        use_dim_redux : bool, default True
            Apply SVD reduction.
        redux_n_components : int | dict[int,int], default 1
            Global component count or per-k mapping when grouped.
        use_tfidf : bool, default True
            Apply TF-IDF weighting.
        sparse : bool, default True
            Return transformed result as sparse matrix (otherwise DataFrame).
        group_kmer_redux_by_length : bool, default True
            Use per-k SVD decomposition.

        Returns
        -------
        pandas.DataFrame | scipy.sparse.csr_matrix
            Transformed representation (type governed by ``sparse`` parameter).
        Notes
        -----
        The transformation pipeline mirrors in-memory handling performed in
        ``aggregate_features`` when raw matrices are supplied directly.
        """
        try:
            from utils.kmer_redux import (
                load_kmer_results,
                apply_kmer_transformations as _apply,
            )
        except ImportError:  # pragma: no cover
            raise RuntimeError(
                "kmer_redux utilities not available to load k-mer paths."
            )

        sparse_matrix, row_names, col_names = load_kmer_results(
            base_path,
            redux_n_components
            if isinstance(redux_n_components, dict)
            else int(redux_n_components),
            redux=False,  # perform redux via unified path below
            group_redux_kmer_len=group_kmer_redux_by_length,
            tfidf=False,  # handle tfidf in unified logic
            verbose=True,
        )
        if sparse_matrix is None:
            raise RuntimeError(
                f"Failed to load k-mer artefacts from base path '{base_path}'."
            )

        transformed_obj, transformed_names = _apply(
            sparse_matrix,
            row_names,
            col_names,
            use_dim_redux=use_dim_redux,
            redux_n_components=redux_n_components,
            use_tfidf=use_tfidf,
            sparse=sparse,
            group_redux_kmer_len=group_kmer_redux_by_length,
        )
        if sparse and not isinstance(transformed_obj, pd.DataFrame):
            return transformed_obj
        if not isinstance(transformed_obj, pd.DataFrame):
            transformed_obj = pd.DataFrame(
                transformed_obj, index=row_names, columns=transformed_names
            )
        return transformed_obj

    def run_all(
        self,
        gtf_file: str,
        ref_genome_path: str,
        config_file: str,
        bed_file: Optional[str] = None,
        fasta_file: Optional[str] = None,
        output_file: Optional[str] = None,
        k_min: int = 3,
        k_max: int = 12,
        use_dim_redux: bool = True,
        redux_n_components: int = 1,
        use_tfidf: bool = True,
        sparse: bool = True,
        group_kmer_redux_by_length: bool = True,
        kmer_sparse_base: Optional[str] = None,
        use_saved_kmer_base: Optional[str] = None,
        return_kmer_sparse_paths: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """Full pipeline execution with unified aggregation.

        Parameters
        ----------
        gtf_file : str
            GTF annotation file (required). Used to extract properly spliced sequences
            and auto-generate BED file for BWQ if not provided.
        ref_genome_path : str
            Reference genome FASTA file for sequence extraction.
        config_file : str
            BigWig/BigBed configuration for ``bwq``.
        bed_file : str, optional
            Transcript regions for BWQ. Auto-generated from GTF if not provided.
        fasta_file : str, optional
            Pre-existing FASTA to reuse. If omitted, sequences are extracted from GTF.
        output_file : str, optional
            Where to persist final parquet feature table.
        k_min, k_max : int, default 3..12
            Inclusive k-mer size bounds.
        use_dim_redux : bool, default True
            Enable SVD dimensionality reduction for k-mer space.
        redux_n_components : int, default 1
            Truncated SVD target rank (per group if grouped mode).
        use_tfidf : bool, default True
            Apply TF-IDF weighting to k-mer counts (before SVD if enabled).
        sparse : bool, default True
            Retain transformed k-mer representation as sparse matrix stored in
            ``attrs`` rather than expanding columns.
        group_kmer_redux_by_length : bool, default True
            If True, independently reduces each k length; else uses one global SVD.
        kmer_sparse_base : str | None, optional
            Base path (no suffix) for saving raw k-mer sparse artefacts
            ("<base>_sparse.npz", "<base>_rows.txt", "<base>_cols.txt"). If
            omitted a path under the wrapper cache is used when persistence is
            requested.
        use_saved_kmer_base : str | None, optional
            When provided, skip k-mer counting and instead load previously
            persisted raw k-mer artefacts (same naming convention as above)
            applying any requested transformations.
        return_kmer_sparse_paths : bool, default False
            If True and k-mers are computed this run, persist raw sparse
            artefacts and attach their paths to ``result_df.attrs['kmer_sparse_paths']``.
        **kwargs : any
            Additional passthrough arguments to individual stage methods.

        Returns
        -------
        pandas.DataFrame
            Unified feature table.

        Side Effects / Attrs
        --------------------
        The returned DataFrame may contain the following entries in
        ``df.attrs``:
        * ``kmer_transformed_sparse`` : transformed sparse matrix (if ``sparse=True``)
        * ``kmer_transformed_names``  : list of reduced / transformed feature names
        * ``kmer_sparse_paths``       : dict of persisted raw k-mer artefact file paths

        Raises
        ------
        FileNotFoundError
            If mandatory upstream files are missing (GTF, genome FASTA, config).
        RuntimeError
            If any stage irrecoverably fails or no sequences are retrievable.
        """
        logger.info("Running all feature extraction scripts...")

        # Validate GTF file exists
        if not os.path.exists(gtf_file):
            raise FileNotFoundError(f"GTF file not found: {gtf_file}")

        # Set up composite progress tracking for all phases
        progress_manager = get_progress_manager(
            show_progress=self.show_progress, quiet=self.quiet
        )

        # Create main progress bar for all feature extraction phases
        main_progress = progress_manager.create_bar(
            total=3,  # BWQ, MFE, K-mer (sequence extraction is preparatory)
            desc="Feature extraction",
            unit="modules",
        )

        # Create temporary directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate BED from GTF if not provided
            if bed_file is None:
                bed_file = os.path.join(temp_dir, "transcripts_from_gtf.bed")
                self.gtf_transcripts_to_bed(gtf_file, bed_file)
                logger.info(f"Generated BED from GTF: {bed_file}")

            # Run BigWig Query
            main_progress.set_description("Running BigWig Query")
            bwq_output = os.path.join(temp_dir, "bwq_features.parquet")
            bwq_df = self.run_bwq(
                bed_file=bed_file,
                config_file=config_file,
                output_file=bwq_output,
                **kwargs,
            )
            main_progress.update(1)
            logger.info(f"BigWig Query completed. Shape: {bwq_df.shape}")

            # Acquire sequences for downstream stages
            sequence_df: Optional[pd.DataFrame] = None
            fasta_path: Optional[str] = fasta_file

            if fasta_path:
                logger.info(f"Loading sequences from provided FASTA: {fasta_path}")
                sequence_df = self._load_fasta_to_dataframe(fasta_path)
            else:
                # Extract sequences from GTF (always uses GTF-based extraction)
                fasta_output = os.path.join(temp_dir, "sequences_from_gtf.fasta")
                try:
                    fasta_path, sequence_df = self.extract_transcripts_from_gtf(
                        gtf_file=gtf_file,
                        ref_genome_path=ref_genome_path,
                        output_fasta=fasta_output,
                        return_df=True,
                    )  # type: ignore[assignment]
                    logger.info(
                        f"GTF sequence extraction completed. FASTA: {fasta_path} SeqDF shape: {sequence_df.shape if sequence_df is not None else 'None'}"
                    )
                except Exception as e:
                    logger.error(f"GTF sequence extraction failed: {e}")
                    raise RuntimeError(
                        "Unable to extract transcript sequences from GTF"
                    ) from e

            if sequence_df is None or sequence_df.empty:
                raise RuntimeError(
                    "No sequences available for downstream feature generation"
                )

            # Normalise transcript identifier column
            if "transcript_id" not in sequence_df.columns:
                id_col_candidates = ["NAME", "Name", "id", "Id"]
                match = next(
                    (col for col in id_col_candidates if col in sequence_df.columns),
                    None,
                )
                if match:
                    sequence_df = sequence_df[[match, "Sequence"]].rename(
                        columns={match: "transcript_id"}
                    )
                else:
                    logger.warning(
                        "Sequence DataFrame missing explicit identifier; synthesising transcript IDs"
                    )
                    sequence_df = sequence_df.reset_index(drop=True)
                    sequence_df.insert(
                        0,
                        "transcript_id",
                        [f"seq_{i}" for i in range(len(sequence_df))],
                    )
            else:
                sequence_df = sequence_df[["transcript_id", "Sequence"]]

            # Ensure RNA alphabet
            sequence_df["Sequence"] = (
                sequence_df["Sequence"]
                .astype(str)
                .str.upper()
                .str.replace("T", "U", regex=False)
            )

            # Run MFE
            main_progress.set_description("Calculating RNA MFE")
            mfe_output = os.path.join(temp_dir, "mfe_features.parquet")
            mfe_df = self.run_mfe(
                df_input=sequence_df,
                sequence_col="Sequence",
                include_structure=True,
                output_file=mfe_output,
                **kwargs,
            )
            main_progress.update(1)
            logger.info(f"MFE completed. Shape: {mfe_df.shape}")

            kmer_df: Optional[pd.DataFrame] = None
            kmer_paths: Optional[Dict[str, str]] = None
            if use_saved_kmer_base:
                main_progress.set_description("Loading saved k-mer artefacts")
                # Prepare dict of expected paths (only sparse_matrix key strictly needed)
                base = (
                    use_saved_kmer_base[:-12]
                    if use_saved_kmer_base.endswith("_sparse.npz")
                    else use_saved_kmer_base
                )
                kmer_paths = {
                    "sparse_matrix": f"{base}_sparse.npz",
                    "rows": f"{base}_rows.txt",
                    "cols": f"{base}_cols.txt",
                }
                # Validate required file exists
                if not os.path.exists(kmer_paths["sparse_matrix"]):
                    raise FileNotFoundError(
                        f"Saved k-mer sparse matrix not found: {kmer_paths['sparse_matrix']}"
                    )
                main_progress.update(1)
                logger.info(f"Using saved k-mer artefacts base: {base}")
            else:
                # Run K-mer fresh
                main_progress.set_description("Calculating k-mer features")
                kmer_output = os.path.join(temp_dir, "kmer_features.parquet")
                output_format = "sparse_dataframe" if sparse else "dataframe"
                if fasta_path is None:
                    raise RuntimeError(
                        "FASTA path not available for k-mer computation."
                    )
                kmer_result = self.run_kmer(
                    input_path=fasta_path,
                    k_min=k_min,
                    k_max=k_max,
                    output_file=kmer_output,
                    return_sparse_paths=return_kmer_sparse_paths,
                    sparse_base_name=kmer_sparse_base,
                    output_format=output_format,
                    **kwargs,
                )
                if return_kmer_sparse_paths and isinstance(kmer_result, tuple):
                    kmer_df, kmer_paths = kmer_result
                else:
                    kmer_df = (
                        kmer_result
                        if not isinstance(kmer_result, tuple)
                        else kmer_result[0]
                    )
                main_progress.update(1)
                if kmer_df is not None and hasattr(kmer_df, "shape"):
                    logger.info(f"K-mer completed. Shape: {kmer_df.shape}")
                else:
                    logger.info("K-mer completed.")

            # Aggregate features
            result_df = self.aggregate_features(
                bed_file=bed_file,
                bwq_df=bwq_df,
                mfe_df=mfe_df,
                kmer_df=kmer_df,
                kmer_sparse_paths=kmer_paths,
                use_dim_redux=use_dim_redux,
                redux_n_components=redux_n_components,
                use_tfidf=use_tfidf,
                sparse=sparse,
                group_kmer_redux_by_length=group_kmer_redux_by_length,
            )

            if kmer_paths and return_kmer_sparse_paths:
                result_df.attrs["kmer_sparse_paths"] = kmer_paths

            # Save output if requested
            if output_file:
                logger.info(f"Saving aggregated features to {output_file}")
                result_df.to_parquet(output_file)

            main_progress.set_description("Feature extraction complete")
            main_progress.close()
            return result_df

    def aggregate_features(
        self,
        bed_file: Optional[str] = None,
        bwq_df: Optional[pd.DataFrame] = None,
        mfe_df: Optional[pd.DataFrame] = None,
        kmer_df: Optional[Union[pd.DataFrame, sparse.csr_matrix]] = None,
        kmer_sparse_paths: Optional[Dict[str, str]] = None,
        id_col: str = "transcript_id",
        use_dim_redux: bool = True,
        redux_n_components: int = 1,
        use_tfidf: bool = True,
        sparse: bool = True,
        group_kmer_redux_by_length: bool = True,
    ) -> pd.DataFrame:
        """Merge individual feature DataFrames into a unified table.

        K-mer transformation (TF-IDF / SVD) is applied here if a raw sparse
        matrix is supplied and transformation utilities are available. Sparse
        results are embedded in ``attrs`` when requested to avoid column bloat.

        Parameters
        ----------
        bed_file : str, optional
            BED file path; when present its ``Name`` column seeds the join index.
        bwq_df, mfe_df : pandas.DataFrame
            Mandatory feature provider outputs.
        kmer_df : pandas.DataFrame | scipy.sparse.csr_matrix, optional
            Raw k-mer counts (dense sparse-aware DataFrame or CSR matrix) produced
            in the current session.
        kmer_sparse_paths : dict[str,str], optional
            Alternative to ``kmer_df``; dictionary containing at minimum the
            key ``'sparse_matrix'`` pointing to a ``*_sparse.npz`` file plus
            accompanying ``*_rows.txt`` / ``*_cols.txt``. When provided the
            raw artefacts are loaded and transformed (TF-IDF / SVD) without
            re-computation.
        id_col : str, default "transcript_id"
            Primary key for joins. Falls back to ``transcript_id`` if not found.
        use_dim_redux : bool, default True
            Enable k-mer dimensionality reduction.
        redux_n_components : int, default 1
            Target dimension for SVD (per group when grouped).
        use_tfidf : bool, default True
            Apply TF-IDF scaling to raw counts.
        sparse : bool, default True
            Retain transformed k-mer representation sparsely (stored in attrs).
        group_kmer_redux_by_length : bool, default True
            Grouped vs global SVD strategy.

        Returns
        -------
        pandas.DataFrame
            Aggregated feature matrix with consistent key column restored.

        Raises
        ------
        ValueError
            If required non-k-mer feature DataFrames are absent or neither
            ``kmer_df`` nor ``kmer_sparse_paths`` is supplied.

        Notes
        -----
        The transformation sequence for k-mers is: optional TF-IDF → optional
        grouped/global Truncated SVD. When ``sparse=True`` the resulting matrix
        and feature names are stored in ``attrs`` rather than expanded into
        columns. This prevents excessive memory use for large k ranges.
        """
        logger.info("Aggregating features...")

        # Validate mandatory components; k-mer can be deferred via kmer_sparse_paths
        mandatory_pairs = [
            (bwq_df, "BigWig Query"),
            (mfe_df, "MFE"),
        ]
        missing = [name for df, name in mandatory_pairs if df is None]
        if missing:
            raise ValueError(
                f"Missing required feature DataFrames: {', '.join(missing)}"
            )
        if kmer_df is None and kmer_sparse_paths is None:
            raise ValueError(
                "Either kmer_df or kmer_sparse_paths must be supplied for k-mer features."
            )

        # Check if id_col exists in tabular (non-sparse-matrix) DataFrames
        try:
            import scipy.sparse as sp  # type: ignore
        except Exception:  # pragma: no cover

            class Dummy:
                @staticmethod
                def issparse(x):
                    return False

            sp = Dummy()  # type: ignore

        for cand, name in [
            (bwq_df, "BigWig Query"),
            (mfe_df, "MFE"),
            (kmer_df, "K-mer"),
        ]:
            if cand is None:
                continue
            if sp.issparse(cand):
                continue
            if id_col not in cand.columns and "transcript_id" not in cand.columns:
                logger.warning(
                    f"{id_col} column not found in {name} DataFrame. Merging may fail."
                )

        # Start from BED-derived identifiers when available to retain positional columns
        if bed_file and os.path.exists(bed_file):
            bed_ranges = pr.read_bed(bed_file)
            bed_frame = bed_ranges.df.copy()
            if id_col in bed_frame.columns:
                pass
            elif "Name" in bed_frame.columns:
                bed_frame = bed_frame.rename(columns={"Name": id_col})
            elif "transcript_id" in bed_frame.columns:
                bed_frame = bed_frame.rename(columns={"transcript_id": id_col})
            else:
                raise ValueError(f"{id_col} not found in BED file")

            # Compute canonical length prior to renaming other columns
            start_col = next(
                (c for c in ["Start", "start"] if c in bed_frame.columns), None
            )
            end_col = next((c for c in ["End", "end"] if c in bed_frame.columns), None)
            if start_col and end_col and "length" not in bed_frame.columns:
                bed_frame["length"] = bed_frame[end_col] - bed_frame[start_col]

            # Prefix remaining BED columns to avoid collisions downstream
            rename_map = {}
            for col in bed_frame.columns:
                if col in {id_col, "length"}:
                    continue
                rename_map[col] = f"bed_{col.lower()}"
            if rename_map:
                bed_frame = bed_frame.rename(columns=rename_map)

            # Retain only unique transcript entries to avoid duplicate indices
            df = bed_frame.drop_duplicates(subset=[id_col])
        else:
            logger.warning(
                "BED file not provided or not found. Using BigWig DataFrame as base."
            )
            if bwq_df is None:
                raise ValueError(
                    "Cannot infer transcript identifiers without BED file or BigWig DataFrame"
                )
            use_col = id_col if id_col in bwq_df.columns else "transcript_id"
            if use_col not in bwq_df.columns:
                raise ValueError(f"{id_col} not found in BigWig DataFrame")
            base_ids = bwq_df[[use_col]].copy()
            if use_col != id_col:
                base_ids = base_ids.rename(columns={use_col: id_col})
            df = base_ids.drop_duplicates(subset=[id_col])

        df.set_index(id_col, inplace=True)

        # Merge BigWig Query features

        # Set index if needed
        assert bwq_df is not None
        if id_col in bwq_df.columns:
            bwq_df = bwq_df.set_index(id_col, drop=True)
        elif "transcript_id" in bwq_df.columns and id_col != "transcript_id":
            bwq_df = bwq_df.rename(columns={"transcript_id": id_col}).set_index(
                id_col, drop=True
            )

        # Merge with main DataFrame
        df = df.merge(bwq_df, how="inner", left_index=True, right_index=True)

        # Merge MFE features if provided
        assert mfe_df is not None
        mfe_join_col = id_col if id_col in mfe_df.columns else "transcript_id"
        mfe_id_preserved = mfe_df[mfe_join_col]
        mfe_feature_cols = [c for c in mfe_df.columns if c != mfe_join_col]
        mfe_features = mfe_df[mfe_feature_cols].add_prefix("ss_")
        mfe_df = pd.concat([mfe_id_preserved, mfe_features], axis=1)
        if mfe_join_col != id_col:
            mfe_df = mfe_df.rename(columns={mfe_join_col: id_col})
        mfe_df = mfe_df.set_index(id_col, drop=True)

        # Merge with main DataFrame
        df = df.merge(mfe_df, how="inner", left_index=True, right_index=True)

        # Process K-mer features (raw DataFrame, sparse matrix, or saved paths)
        if kmer_sparse_paths is not None and apply_kmer_transformations is not None:
            base_path = kmer_sparse_paths.get("sparse_matrix", "")
            if base_path.endswith("_sparse.npz"):
                base_path = base_path[: -len("_sparse.npz")]  # remove exact suffix
            try:
                transformed = self.load_and_transform_kmer_from_paths(
                    base_path,
                    use_dim_redux=use_dim_redux,
                    redux_n_components=redux_n_components,
                    use_tfidf=use_tfidf,
                    sparse=sparse,
                    group_kmer_redux_by_length=group_kmer_redux_by_length,
                )
                if sparse and not isinstance(transformed, pd.DataFrame):
                    df.attrs["kmer_transformed_sparse"] = transformed
                else:
                    if not isinstance(transformed, pd.DataFrame):
                        transformed = pd.DataFrame(
                            transformed.toarray(), index=df.index
                        )
                    if transformed.index.name != df.index.name:
                        transformed.index = df.index
                    df = df.merge(
                        transformed, how="inner", left_index=True, right_index=True
                    )
            except Exception as e:
                logger.error(f"Failed loading k-mer sparse artifacts: {e}")
        elif kmer_df is not None:
            if isinstance(kmer_df, pd.DataFrame):
                kmer_work = kmer_df.copy()
                if id_col in kmer_work.columns:
                    kmer_work.set_index(id_col, inplace=True)
                elif "transcript_id" in kmer_work.columns and id_col != "transcript_id":
                    kmer_work.rename(columns={"transcript_id": id_col}, inplace=True)
                    kmer_work.set_index(id_col, inplace=True)
                else:
                    # Fall back to using the existing index as transcript identifier when aligned
                    if kmer_work.index.name in {None, "index"} and len(
                        kmer_work.index
                    ) == len(df.index):
                        kmer_work.index = pd.Index(df.index, name=id_col)
                    elif (
                        kmer_work.index.name == "transcript_id"
                        and id_col != "transcript_id"
                    ):
                        kmer_work.index.name = id_col
                if apply_kmer_transformations is not None and (
                    use_dim_redux or use_tfidf
                ):
                    from scipy.sparse import csr_matrix

                    csr = csr_matrix(
                        kmer_work.sparse.to_coo()
                        if hasattr(kmer_work, "sparse")
                        else kmer_work.to_numpy()
                    )
                    ids = df.index.tolist()
                    kmer_names = [str(c) for c in kmer_work.columns]
                    transformed_obj, transformed_names = apply_kmer_transformations(
                        csr,
                        ids,
                        kmer_names,
                        use_dim_redux=use_dim_redux,
                        redux_n_components=redux_n_components,
                        use_tfidf=use_tfidf,
                        sparse=sparse,
                        group_redux_kmer_len=group_kmer_redux_by_length,
                    )
                    if sparse and not isinstance(transformed_obj, pd.DataFrame):
                        df.attrs["kmer_transformed_sparse"] = transformed_obj
                        df.attrs["kmer_transformed_names"] = transformed_names
                    else:
                        if not isinstance(transformed_obj, pd.DataFrame):
                            transformed_obj = pd.DataFrame(
                                transformed_obj,
                                index=df.index,
                                columns=transformed_names,
                            )
                        df = df.merge(
                            transformed_obj,
                            how="inner",
                            left_index=True,
                            right_index=True,
                        )
                else:
                    df = df.merge(
                        kmer_work, how="inner", left_index=True, right_index=True
                    )
            else:
                if apply_kmer_transformations is None:
                    logger.warning(
                        "kmer_redux utilities not available; skipping k-mer transformations."
                    )
                    if sparse:
                        df.attrs["kmer_sparse"] = kmer_df
                    else:
                        dense = kmer_df.toarray()
                        dense_df = pd.DataFrame(
                            dense,
                            index=df.index,
                            columns=[f"kmer_{i}" for i in range(dense.shape[1])],
                        )
                        df = df.merge(
                            dense_df, how="inner", left_index=True, right_index=True
                        )
                else:
                    ids = df.index.tolist()
                    kmer_names = [f"kmer_{i}" for i in range(kmer_df.shape[1])]
                    transformed_obj, transformed_names = apply_kmer_transformations(
                        kmer_df,
                        ids,
                        kmer_names,
                        use_dim_redux=use_dim_redux,
                        redux_n_components=redux_n_components,
                        use_tfidf=use_tfidf,
                        sparse=sparse,
                        group_redux_kmer_len=group_kmer_redux_by_length,
                    )
                    if sparse and not isinstance(transformed_obj, pd.DataFrame):
                        df.attrs["kmer_transformed_sparse"] = transformed_obj
                        df.attrs["kmer_transformed_names"] = transformed_names
                    else:
                        if not isinstance(transformed_obj, pd.DataFrame):
                            transformed_obj = pd.DataFrame(
                                transformed_obj,
                                index=ids,
                                columns=transformed_names,
                            )
                        df = df.merge(
                            transformed_obj,
                            how="inner",
                            left_index=True,
                            right_index=True,
                        )

        # Reset index to make the primary key a column again
        df.reset_index(inplace=True)

        # Ensure length column present for downstream consumption
        if "length" not in df.columns:
            start_candidates = ["Start", "start", "bed_start"]
            end_candidates = ["End", "end", "bed_end"]
            start_col = next((c for c in start_candidates if c in df.columns), None)
            end_col = next((c for c in end_candidates if c in df.columns), None)
            if start_col and end_col:
                df["length"] = df[end_col] - df[start_col]
            else:
                raise ValueError(
                    "Start and End columns not found for length calculation."
                )

        # Remove any residual duplicate columns that may have slipped through
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]

        # Drop all BED-related columns (prefixed with 'bed_') except length and id_col
        bed_cols_to_drop = [col for col in df.columns if str(col).startswith("bed_")]
        if bed_cols_to_drop:
            logger.info(
                f"Dropping {len(bed_cols_to_drop)} BED coordinate columns: {bed_cols_to_drop}"
            )
            df = df.drop(columns=bed_cols_to_drop)

        logger.info(f"Aggregated feature DataFrame shape: {df.shape}")

        df.columns = [str(col).lower() for col in df.columns]

        return df


def main():
    """CLI entry point.

    Provides subcommands for individual feature modules (``bwq``, ``mfe``,
    ``kmer``) plus ``all`` (full pipeline) and ``cache`` management.
    Argument parsing delegates to corresponding wrapper methods. Exit codes are
    propagated from unhandled exceptions; normal completion prints shape info
    for executed tasks.
    """
    parser = argparse.ArgumentParser(
        description="Unified interface for running feature extraction."
    )

    # Common arguments
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of worker threads/processes to use across feature generators",
    )

    # Caching arguments
    parser.add_argument(
        "--cache-dir",
        help="Directory for caching downloaded files (default: ./bwq_tracks/bwq_persistent_cache)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable file caching (files will be downloaded every time)",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear the cache before running"
    )

    # Add progress-related CLI arguments
    update_cli_args(parser)

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # BigWig Query command
    bwq_parser = subparsers.add_parser(
        "bwq", help="Run BigWig Query feature extraction"
    )
    bwq_parser.add_argument(
        "--bed", required=True, help="Path to BED file containing regions to query"
    )
    bwq_parser.add_argument(
        "--bwq-config",
        required=True,
        help="Path to YAML configuration file specifying BigWig/BigBed files and statistics",
    )
    bwq_parser.add_argument(
        "--output", "-o", help="Output file path for the results (parquet format)"
    )

    # MFE command
    mfe_parser = subparsers.add_parser("mfe", help="Run MFE feature extraction")
    mfe_parser.add_argument(
        "--input",
        required=True,
        help="Path to input DataFrame with sequences (parquet format)",
    )
    mfe_parser.add_argument(
        "--sequence-col",
        default="Sequence",
        help="Column name containing RNA sequences",
    )
    mfe_parser.add_argument(
        "--include-structure",
        action="store_true",
        help="Include predicted RNA secondary structure in output",
    )
    mfe_parser.add_argument(
        "--num-processes",
        type=int,
        help="Number of processes to use for parallel computation",
    )
    mfe_parser.add_argument(
        "--output", "-o", help="Output file path for the results (parquet format)"
    )

    # K-mer command
    kmer_parser = subparsers.add_parser("kmer", help="Run K-mer feature extraction")
    kmer_parser.add_argument(
        "--input",
        required=True,
        help="Path to FASTA file or directory containing FASTA files",
    )
    kmer_parser.add_argument(
        "--k-min", type=int, default=3, help="Minimum k-mer length"
    )
    kmer_parser.add_argument(
        "--k-max", type=int, default=12, help="Maximum k-mer length"
    )
    kmer_parser.add_argument(
        "--output-format",
        choices=["dataframe", "sparse_dataframe", "matrix"],
        default="dataframe",
        help="Output format",
    )
    kmer_parser.add_argument(
        "--output", "-o", help="Output file path for the results (parquet format)"
    )
    kmer_parser.add_argument(
        "--return-sparse-paths",
        action="store_true",
        help="Persist and return raw sparse k-mer artefact paths",
    )
    kmer_parser.add_argument(
        "--sparse-base", help="Base path (no suffix) to save sparse k-mer artefacts"
    )

    # Run all command
    all_parser = subparsers.add_parser("all", help="Run all feature extraction scripts")
    all_parser.add_argument(
        "--gtf",
        required=True,
        help="Path to GTF annotation file (required for sequence extraction and BED generation)",
    )
    all_parser.add_argument(
        "--bed",
        help="Path to BED file containing regions to query (optional, auto-generated from GTF if not provided)",
    )
    all_parser.add_argument(
        "--fasta",
        help="Path to pre-extracted FASTA sequences (optional, auto-extracted from GTF if not provided)",
    )
    all_parser.add_argument(
        "--bwq-config",
        required=True,
        help="Path to YAML configuration file specifying BigWig/BigBed files and statistics",
    )
    all_parser.add_argument(
        "--ref-genome", required=True, help="Path to reference genome FASTA file"
    )
    all_parser.add_argument("--k-min", type=int, default=3, help="Minimum k-mer length")
    all_parser.add_argument(
        "--k-max", type=int, default=12, help="Maximum k-mer length"
    )
    all_parser.add_argument(
        "--use-dim-redux",
        action="store_true",
        help="Use dimensionality reduction for k-mer features",
    )
    all_parser.add_argument(
        "--redux-n-components",
        type=int,
        default=1,
        help="Number of components for dimensionality reduction",
    )
    all_parser.add_argument(
        "--use-tfidf",
        action="store_true",
        help="Apply TF-IDF transformation to k-mer features",
    )
    all_parser.add_argument(
        "--sparse", action="store_true", help="Keep k-mer features as sparse matrix"
    )
    all_parser.add_argument(
        "--no-group-kmer-by-length",
        action="store_true",
        help="Disable grouped SVD per k-mer length (use single global SVD).",
    )
    all_parser.add_argument(
        "--output", "-o", help="Output file path for the results (parquet format)"
    )
    all_parser.add_argument(
        "--kmer-sparse-base",
        help="Base path (no suffix) to save raw k-mer sparse artefacts",
    )
    all_parser.add_argument(
        "--use-saved-kmer-base",
        help="Base path (no suffix) of previously saved raw k-mer artefacts to reuse",
    )
    all_parser.add_argument(
        "--return-kmer-sparse-paths",
        action="store_true",
        help="Return and attach raw k-mer sparse artefact paths",
    )

    # Cache management command
    cache_parser = subparsers.add_parser("cache", help="Manage file cache")
    cache_subparsers = cache_parser.add_subparsers(
        dest="cache_action", help="Cache action"
    )

    # Cache info command
    cache_info_parser = cache_subparsers.add_parser(
        "info", help="Show cache information"
    )

    # Cache clear command
    cache_clear_parser = cache_subparsers.add_parser("clear", help="Clear the cache")

    args = parser.parse_args()

    # Resolve progress settings from CLI arguments
    progress_settings = resolve_progress_settings(args)

    # Initialize the wrapper with the specified parameters
    wrapper = FeatureWrapper(
        log_level=args.log_level,
        cache_dir=args.cache_dir,
        keep_downloaded_files=not args.no_cache,
        clear_cache_on_startup=args.clear_cache,
        threads=args.threads,
        **progress_settings,
    )

    # Process arguments and run appropriate functions
    if args.command == "cache":
        if args.cache_action == "info":
            cache_info = wrapper.get_cache_info()
            print("\n=== Cache Information ===")
            for key, value in cache_info.items():
                if key == "cached_files" and isinstance(value, list):
                    print(f"{key}: {len(value)} files")
                    if value:
                        preview = value[:5]
                        for filename in preview:
                            print(f"  - {filename}")
                        remaining = len(value) - len(preview)
                        if remaining > 0:
                            print(f"  ... and {remaining} more files")
                else:
                    print(f"{key}: {value}")
        elif args.cache_action == "clear":
            wrapper.clear_cache()
            print("Cache cleared successfully")
        else:
            cache_parser.print_help()
        return

    elif args.command == "bwq":
        result = wrapper.run_bwq(
            bed_file=args.bed,
            config_file=args.bwq_config,
            output_file=args.output,
            threads=args.threads,
        )
        logger.info(f"BigWig Query completed. Shape: {result.shape}")

    elif args.command == "mfe":
        # Load input DataFrame
        df_input = pd.read_parquet(args.input)
        num_processes = (
            args.num_processes if args.num_processes is not None else args.threads
        )
        result = wrapper.run_mfe(
            df_input=df_input,
            sequence_col=args.sequence_col,
            include_structure=args.include_structure,
            num_processes=num_processes,
            output_file=args.output,
        )
        logger.info(f"MFE completed. Shape: {result.shape}")

    elif args.command == "kmer":
        result = wrapper.run_kmer(
            input_path=args.input,
            k_min=args.k_min,
            k_max=args.k_max,
            output_format=args.output_format,
            output_file=args.output,
            return_sparse_paths=args.return_sparse_paths,
            sparse_base_name=args.sparse_base,
            num_workers=args.threads,
        )
        if isinstance(result, tuple):
            logger.info(
                f"K-mer completed. Shape: {result[0].shape}; paths: {result[1]}"
            )
        else:
            logger.info(f"K-mer completed. Shape: {result.shape}")

    elif args.command == "all":
        result = wrapper.run_all(
            gtf_file=args.gtf,
            bed_file=args.bed,
            fasta_file=args.fasta,
            config_file=args.bwq_config,
            ref_genome_path=args.ref_genome,
            k_min=args.k_min,
            k_max=args.k_max,
            use_dim_redux=args.use_dim_redux,
            redux_n_components=args.redux_n_components,
            use_tfidf=args.use_tfidf,
            sparse=args.sparse,
            group_kmer_redux_by_length=not args.no_group_kmer_by_length,
            output_file=args.output,
            kmer_sparse_base=args.kmer_sparse_base,
            use_saved_kmer_base=args.use_saved_kmer_base,
            return_kmer_sparse_paths=args.return_kmer_sparse_paths,
            threads=args.threads,
        )
        logger.info(f"All feature extraction completed. Shape: {result.shape}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
