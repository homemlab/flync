import argparse
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import yaml

# Import progress utilities
try:
    from src.utils.progress import get_progress_manager, update_cli_args, resolve_progress_settings
except ImportError:
    # Try relative import when running as script
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.progress import get_progress_manager, update_cli_args, resolve_progress_settings


# --- File Manager Class ---
class BigWigFileManager:
    """
    Manages BigWig/BigBed file handles for efficient access, including local caching of remote files.

    This class handles opening local and remote files, downloading remote files
    with retries, caching downloaded files (persistently or transiently),
    and managing file handles to avoid repeatedly opening the same file.
    It uses an ExitStack to ensure proper cleanup of resources.
    """

    def __init__(
        self,
        url_timeout=30,
        max_retries=3,
        base_storage_directory=None,
        verify_ssl=True,
        keep_downloaded_files=False,
    ):
        """
        Initializes the BigWigFileManager.

        Args:
            url_timeout (int): Timeout in seconds for URL connections. Defaults to 30.
            max_retries (int): Maximum number of retry attempts for URL connections. Defaults to 3.
            base_storage_directory (str, optional): Base directory for storing cached and temporary files.
                                                     Defaults to the system's temporary directory.
            verify_ssl (bool): Whether to verify SSL certificates for HTTPS URLs. Defaults to True.
            keep_downloaded_files (bool): If True, downloaded remote files are kept in a persistent cache
                                          directory ({base_storage_directory}/bwq_persistent_cache).
                                          If False, they are stored in a transient directory and deleted
                                          on exit. Defaults to False.
        """
        self.file_handles = {}
        self.exit_stack = ExitStack()
        self.url_timeout = url_timeout
        self.max_retries = max_retries

        self.base_dir = base_storage_directory or tempfile.gettempdir()
        self.persistent_cache_dir = os.path.join(self.base_dir, "bwq_persistent_cache")
        self.transient_temp_dir = os.path.join(self.base_dir, "bwq_transient_temp")

        os.makedirs(self.persistent_cache_dir, exist_ok=True)
        os.makedirs(self.transient_temp_dir, exist_ok=True)

        self.verify_ssl = verify_ssl
        self.keep_downloaded_files = keep_downloaded_files
        self.transient_files_to_delete = []  # Track transient temporary files for cleanup
        self.downloading = set()  # Track URLs that are currently being downloaded
        self.download_lock = threading.Lock()  # Lock for thread-safe downloading
        self.handle_lock = threading.Lock()  # Lock for thread-safe file handle access

    def _get_persistent_cache_path(self, url):
        """Generates a predictable, safe file path within the persistent cache directory for a given URL."""
        # Using quote_plus for filename safety, and a prefix
        safe_filename = "bwq_cache_" + urllib.parse.quote_plus(url)
        return os.path.join(self.persistent_cache_dir, safe_filename)

    def _perform_download(self, url, target_local_path):
        """
        Downloads a remote file to a specified local path with retries.

        Args:
            url (str): The URL of the file to download.
            target_local_path (str): The local file path where the downloaded file should be saved.

        Returns:
            str: The path to the downloaded file (target_local_path) if successful, None otherwise.
        """
        headers = {
            "User-Agent": "BigWigQuery/1.1",  # Updated version
        }
        context = None
        if not self.verify_ssl:
            import ssl

            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        retry_count = 0
        while retry_count < self.max_retries:
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(
                    req, timeout=self.url_timeout, context=context
                ) as response:
                    with open(target_local_path, "wb") as out_file:
                        shutil.copyfileobj(response, out_file)
                logging.info(f"Successfully downloaded {url} to {target_local_path}")
                return target_local_path
            except Exception as e:
                retry_count += 1
                if retry_count < self.max_retries:
                    sleep_time = 2**retry_count
                    logging.warning(
                        f"Attempt {retry_count} failed for {url} to {target_local_path}: {e}. "
                        f"Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                else:
                    logging.error(
                        f"Failed to download {url} to {target_local_path} after {self.max_retries} attempts: {e}"
                    )
                    # Clean up partially downloaded file if it exists
                    if os.path.exists(target_local_path):
                        try:
                            os.unlink(target_local_path)
                        except Exception as unlink_e:
                            logging.warning(
                                f"Could not remove partially downloaded file {target_local_path}: {unlink_e}"
                            )
                    return None
        return None

    def get_file_handle(self, file_path):
        """
        Retrieves an existing or opens a new file handle for a local path or URL.

        Handles downloading and caching for URLs based on initialization settings.
        Uses locks to ensure thread safety when multiple threads request the same file.

        Args:
            file_path (str): The local file path or URL of the BigWig/BigBed file.

        Returns:
            pyBigWig file handle: An open handle to the requested file.
            None: If the file cannot be accessed, downloaded, or opened.
        """
        # First check if we already have this file handle (quick check with lock)
        with self.handle_lock:
            if file_path in self.file_handles:
                logging.debug(f"Using existing file handle for {file_path}")
                return self.file_handles[file_path]

        # Handle is not available, so we need to open the file
        parsed_url = urllib.parse.urlparse(file_path)
        is_url = parsed_url.scheme in ("http", "https", "ftp")

        if is_url:
            persistent_path = self._get_persistent_cache_path(file_path)
            # If persistent cache file exists, open it directly (skip waiting for download)
            if os.path.exists(persistent_path):
                try:
                    bw = self.exit_stack.enter_context(pyBigWig.open(persistent_path))
                    with self.handle_lock:
                        self.file_handles[file_path] = bw
                    logging.debug(
                        f"Using existing persistent cached file: {persistent_path} for {file_path}"
                    )
                    return bw
                except Exception as e:
                    logging.error(
                        f"Error opening cached file {persistent_path} for {file_path}: {e}"
                    )
                    # If cache file is corrupted, fallback to download logic below

            # For URLs, we need to handle downloading with thread safety
            with self.download_lock:
                if (
                    file_path in self.file_handles
                ):  # Double check, might have been opened by another thread
                    return self.file_handles[file_path]
                if file_path in self.downloading:
                    logging.debug(
                        f"Waiting for {file_path} to be downloaded by another thread (initial check)"
                    )
                    # Fall through to wait logic outside this lock
                else:
                    # Mark this URL as being downloaded by THIS thread if not already marked
                    self.downloading.add(file_path)

            # Wait for the file to be downloaded by another thread if it was marked by another thread
            # This loop handles the case where another thread started the download just before this one.
            is_being_downloaded_by_other = False
            with self.download_lock:  # Check self.downloading under lock
                if file_path not in self.file_handles and file_path in self.downloading:
                    # If it's in self.downloading but not yet in self.file_handles,
                    # and this thread is NOT the one that added it (implicit: means another thread did)
                    # This logic is tricky because we don't store *which* thread is downloading.
                    # A simpler approach might be to always wait if in self.downloading and not in self.file_handles.
                    # For now, assume if it's in downloading, we wait.
                    is_being_downloaded_by_other = True

            if is_being_downloaded_by_other:
                max_wait_attempts = 60  # e.g., 30 seconds if sleep is 0.5s
                for attempt in range(max_wait_attempts):
                    # If persistent cache file now exists (created by another thread), open it directly
                    if os.path.exists(persistent_path):
                        try:
                            bw = self.exit_stack.enter_context(pyBigWig.open(persistent_path))
                            with self.handle_lock:
                                self.file_handles[file_path] = bw
                            logging.debug(
                                f"Using existing persistent cached file: {persistent_path} for {file_path}"
                            )
                            return bw
                        except Exception as e:
                            logging.error(
                                f"Error opening cached file {persistent_path} for {file_path}: {e}"
                            )
                            break
                    logging.debug(
                        f"Waiting for {file_path} (attempt {attempt+1}/{max_wait_attempts}) to be downloaded by another thread."
                    )
                    time.sleep(0.5)  # Short wait
                    with self.handle_lock:
                        if file_path in self.file_handles:
                            return self.file_handles[file_path]
                    # Check if the download completed or failed by another thread
                    with self.download_lock:
                        if file_path not in self.downloading and file_path not in self.file_handles:
                            logging.warning(
                                f"Download of {file_path} by another thread seems to have failed or completed without handle. This thread will attempt."
                            )
                            break  # Exit wait loop and attempt download
                else:  # If loop completes without returning
                    logging.warning(
                        f"Timeout waiting for {file_path} download from another thread. This thread will attempt."
                    )
                    # Proceed to attempt download by this thread if timeout occurs

            # Re-check and start downloading if necessary by this thread
            actual_file_to_open = None
            try:
                # Ensure this thread is marked as downloader if it's going to download
                # This re-acquires download_lock to ensure atomicity of check-and-set for downloading status
                with self.download_lock:
                    if (
                        file_path in self.file_handles
                    ):  # Check again, might have been opened while waiting
                        return self.file_handles[file_path]
                    # If not in downloading, this thread claims it. If already in (by this thread), it's fine.
                    self.downloading.add(file_path)

                # Always check persistent cache first, regardless of keep_downloaded_files setting.
                persistent_path = self._get_persistent_cache_path(file_path)
                if os.path.exists(persistent_path):
                    logging.debug(
                        f"Using existing persistent cached file: {persistent_path} for {file_path}"
                    )
                    actual_file_to_open = persistent_path
                elif self.keep_downloaded_files:
                    logging.info(f"Downloading {file_path} to persistent cache: {persistent_path}")
                    actual_file_to_open = self._perform_download(file_path, persistent_path)
                else:  # Not keeping files, and not in cache, so download to a transient temp file
                    temp_fd, temp_path_for_download = tempfile.mkstemp(
                        suffix=os.path.splitext(file_path)[1] or ".bwqtmp",  # Ensure suffix
                        dir=self.transient_temp_dir,
                    )
                    os.close(temp_fd)  # We only need the path for _perform_download

                    logging.info(
                        f"Downloading {file_path} to transient temp file: {temp_path_for_download}"
                    )
                    actual_file_to_open = self._perform_download(file_path, temp_path_for_download)
                    if actual_file_to_open:
                        self.transient_files_to_delete.append(
                            actual_file_to_open
                        )  # Mark for deletion

                if actual_file_to_open is None:
                    logging.error(f"Failed to obtain local file for {file_path}")
                    with self.download_lock:  # Ensure cleanup from downloading set on failure
                        if file_path in self.downloading:
                            self.downloading.remove(file_path)
                    return None

                # Open the local file (cached or temporary)
                bw = self.exit_stack.enter_context(pyBigWig.open(actual_file_to_open))

                with self.handle_lock:
                    self.file_handles[file_path] = bw

                # Remove from downloading set *after* successfully adding to handles
                with self.download_lock:
                    if file_path in self.downloading:
                        self.downloading.remove(file_path)

                logging.info(
                    f"Successfully opened remote BigWig/BigBed file: {file_path} (from {actual_file_to_open})"
                )
                return bw
            except Exception as e:
                logging.error(f"Error opening or processing remote file {file_path}: {e}")
                with self.download_lock:  # Ensure cleanup from downloading set on exception
                    if file_path in self.downloading:
                        self.downloading.remove(file_path)
                return None
        else:
            # Regular local file - simpler handling
            try:
                bw = self.exit_stack.enter_context(pyBigWig.open(file_path))
                with self.handle_lock:
                    self.file_handles[file_path] = bw
                return bw
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                return None

    def preload_remote_files(self, file_paths, progress_bar=None):
        """
        Attempts to download and open handles for all unique remote URLs in the given list.

        This is useful to ensure remote files are available and cached before starting
        parallel processing that might request them concurrently.

        Args:
            file_paths (list[str]): A list of file paths or URLs. Only URLs will be preloaded.
            progress_bar: Optional progress bar to update during preloading.

        Returns:
            int: The number of remote files successfully preloaded (downloaded and opened).
        """
        remote_urls = set()

        # Find unique remote URLs
        for path in file_paths:
            parsed_url = urllib.parse.urlparse(path)
            is_url = parsed_url.scheme in ("http", "https", "ftp")
            if is_url:
                remote_urls.add(path)

        # Download each remote URL
        success_count = 0
        if not remote_urls:
            logging.info("No remote files to preload.")
            return 0

        logging.info(f"Attempting to preload {len(remote_urls)} remote files...")

        # Parallel preloading for better performance
        if len(remote_urls) > 1:
            max_preload_workers = min(len(remote_urls), 4)  # Limit concurrent downloads
            with ThreadPoolExecutor(max_workers=max_preload_workers) as preload_executor:
                preload_futures = {
                    preload_executor.submit(self.get_file_handle, url): url for url in remote_urls
                }

                for future in as_completed(preload_futures):
                    url = preload_futures[future]
                    try:
                        if future.result() is not None:
                            success_count += 1
                        else:
                            logging.warning(f"Failed to preload remote file: {url}")
                    except Exception as e:
                        logging.warning(f"Failed to preload remote file {url}: {e}")
                    
                    # Update progress bar if provided
                    if progress_bar:
                        progress_bar.update(1)
                        
        else:
            # Single file - no need for thread pool overhead
            for url in remote_urls:
                if self.get_file_handle(url) is not None:
                    success_count += 1
                else:
                    logging.warning(f"Failed to preload remote file: {url}")
                
                # Update progress bar if provided
                if progress_bar:
                    progress_bar.update(1)

        logging.info(f"Preloaded {success_count}/{len(remote_urls)} remote files")
        return success_count

    def close_all(self):
        """Closes all managed file handles and cleans up any transient temporary files."""
        self.exit_stack.close()
        self.file_handles = {}

        # Clean up any transient temporary files
        for temp_file in self.transient_files_to_delete:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logging.debug(f"Removed transient temporary file: {temp_file}")
            except Exception as e:
                logging.warning(f"Failed to remove transient temporary file {temp_file}: {e}")
        self.transient_files_to_delete = []

    def clear_persistent_cache(self):
        """Removes all files from the persistent cache directory managed by this instance."""
        if not os.path.exists(self.persistent_cache_dir):
            logging.info(
                f"Persistent cache directory {self.persistent_cache_dir} does not exist. Nothing to clear."
            )
            return

        logging.info(f"Clearing persistent cache directory: {self.persistent_cache_dir}")
        cleared_count = 0
        error_count = 0
        for filename in os.listdir(self.persistent_cache_dir):
            file_path = os.path.join(self.persistent_cache_dir, filename)
            try:
                # Ensure we only delete files prefixed by our cache marker
                if filename.startswith("bwq_cache_"):
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        cleared_count += 1
                    elif os.path.isdir(file_path):  # Should not happen with current naming
                        shutil.rmtree(file_path)
                        cleared_count += 1
                else:
                    logging.debug(f"Skipping non-cache file in cache directory: {filename}")
            except Exception as e:
                logging.warning(f"Failed to delete {file_path} from cache: {e}")
                error_count += 1
        if cleared_count > 0 or error_count > 0:
            logging.info(
                f"Cleared {cleared_count} items from persistent cache. Encountered {error_count} errors."
            )
        else:
            logging.info("Persistent cache was empty or contained no matching files.")


# --- Configuration ---
def setup_logging(log_level):
    """
    Configures the root logger for the application.

    Args:
        log_level (str): The desired logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",  # Added module
    )
    logging.info(f"Logging initialized at level: {log_level}")


# --- Helper Functions ---
def is_url(path):
    """
    Checks if a given string path is a URL (http, https, or ftp scheme).

    Args:
        path (str): The path string to check.

    Returns:
        bool: True if the path is a URL, False otherwise.
    """
    parsed = urllib.parse.urlparse(path)
    return parsed.scheme in ("http", "https", "ftp")


def load_config_from_yaml(yaml_path):
    """Loads file configurations from a YAML file with support for multiple stats and name extraction."""
    if not is_url(yaml_path) and not os.path.exists(yaml_path):
        logging.error(f"Configuration file not found: {yaml_path}")
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        # Handle URL for config file
        if is_url(yaml_path):
            with urllib.request.urlopen(yaml_path) as response:
                config_data = yaml.safe_load(response)
        else:
            with open(yaml_path, "r") as f:
                config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file {yaml_path}: {e}")
        raise ValueError(f"Error parsing YAML configuration file {yaml_path}: {e}") from e
    except Exception as e:
        logging.error(f"Error reading configuration file {yaml_path}: {e}")
        raise IOError(f"Error reading configuration file {yaml_path}: {e}") from e

    if not isinstance(config_data, list):
        logging.error(
            f"YAML config should be a list of file configurations, but got {type(config_data)}"
        )
        raise ValueError("YAML config must be a list.")

    expanded_configs = []
    # Added "extract_names" to valid stats
    valid_numerical_stats = ["mean", "max", "min", "coverage", "std", "sum"]
    valid_special_stats = ["extract_names"]

    for i, item in enumerate(config_data):
        if not isinstance(item, dict):
            logging.error(f"Item #{i+1} in config is not a dictionary. Config: {item}")
            raise ValueError(f"Item #{i+1} in config must be a dictionary.")

        if "path" not in item or not isinstance(item["path"], str) or not item["path"]:
            logging.error(f"Item #{i+1} in config has invalid or missing 'path'. Config: {item}")
            raise ValueError(f"Item #{i+1} in config has invalid or missing 'path'.")

        file_path = item["path"]

        # Parse optional range expansion parameters
        upstream = item.get("upstream", 0)  # Default to 0 (no expansion)
        downstream = item.get("downstream", 0)  # Default to 0 (no expansion)

        # Validate expansion parameters
        if not isinstance(upstream, int) or upstream < 0:
            logging.warning(
                f"Invalid 'upstream' value in config item #{i+1}: {upstream}. Must be a non-negative integer. Using 0."
            )
            upstream = 0
        if not isinstance(downstream, int) or downstream < 0:
            logging.warning(
                f"Invalid 'downstream' value in config item #{i+1}: {downstream}. Must be a non-negative integer. Using 0."
            )
            downstream = 0

        # Path existence for local files is checked by file manager later, but good to warn early if possible
        # if not is_url(file_path) and not os.path.exists(file_path):
        #     logging.warning(
        #         f"File specified in config item #{i+1} not found locally: {file_path}. Will attempt to open."
        #     )
        # No longer raising FileNotFoundError here, let BigWigFileManager handle it.

        if "stats" not in item or not isinstance(item["stats"], list):
            logging.error(f"Item #{i+1} 'stats' must be a list. Config: {item}")
            raise ValueError(f"Item #{i+1} 'stats' must be a list.")

        for j, stat_item in enumerate(item["stats"]):
            if not isinstance(stat_item, dict):
                logging.error(
                    f"Stat item #{j+1} in item #{i+1} is not a dictionary. Config: {stat_item}"
                )
                continue  # Skip this malformed stat_item

            if "stat" not in stat_item or "name" not in stat_item:
                logging.error(
                    f"Stat item #{j+1} in item #{i+1} missing 'stat' or 'name'. Config: {stat_item}"
                )
                continue  # Skip

            current_stat = stat_item["stat"]
            current_name = stat_item["name"]
            config_entry = {
                "path": file_path,
                "stat": current_stat,
                "name": current_name,
                "upstream": upstream,
                "downstream": downstream,
            }

            if current_stat in valid_numerical_stats:
                expanded_configs.append(config_entry)
            elif current_stat in valid_special_stats:
                if current_stat == "extract_names":
                    # For extract_names, optionally get name_field_index
                    name_field_index = stat_item.get(
                        "name_field_index", 3
                    )  # Default to 3 (4th field)
                    if not isinstance(name_field_index, int) or name_field_index < 0:
                        logging.error(
                            f"Invalid 'name_field_index' for 'extract_names' in item #{i+1}, stat #{j+1}. "
                            f"Must be a non-negative integer. Got: {name_field_index}. Using default 3."
                        )
                        name_field_index = 3
                    config_entry["name_field_index"] = name_field_index
                expanded_configs.append(config_entry)
            else:
                logging.error(
                    f"Stat item #{j+1} in item #{i+1} has unsupported statistic '{current_stat}'. "
                    f"Supported numerical stats are: {', '.join(valid_numerical_stats)}. "
                    f"Supported special stats are: {', '.join(valid_special_stats)}."
                )
                continue  # Skip unsupported stat

    if not expanded_configs:
        logging.warning(
            f"No valid stat configurations were loaded from {yaml_path}. Output might be empty."
        )
    else:
        logging.info(
            f"Loaded {len(expanded_configs)} stat/extraction configurations from {yaml_path}"
        )
    return expanded_configs


def read_ranges_from_bed(bed_file_path):
    """Reads genomic ranges (chrom, start, end, name) from a BED file."""
    ranges = []
    if not os.path.exists(bed_file_path):
        logging.error(f"BED file not found: {bed_file_path}")
        raise FileNotFoundError(f"BED file not found: {bed_file_path}")

    # Try using pyranges first for robustness, but ensure it gets the name column correctly
    # Pyranges default name column is 'Name'. BED files might not have headers.
    try:
        # Read with standard BED columns, then select what we need.
        # Pyranges might infer column names; if 4th col is name, it often becomes 'Name'.
        gr = pr.read_bed(bed_file_path, as_df=True)  # Read as DataFrame to access columns easily
        if not gr.empty:
            # Ensure required columns exist. Pyranges uses Chromosome, Start, End.
            # The 4th BED column is typically for 'name'. If no header, it might be col_3.
            # If pyranges names it 'Name', use that. Otherwise, try to access by position if possible.
            # For simplicity, we assume pyranges handles BED3/BED4 correctly.
            # If 'Name' column (or equivalent for 4th field) is not present, it will be an issue.
            # The manual parser below is a fallback.

            # Standard BED columns are 0:chrom, 1:start, 2:end, 3:name
            # Pyranges DataFrame usually has 'Chromosome', 'Start', 'End'.
            # The 4th column (name) might be 'Name' or an unnamed column if no header.
            # If 'Name' column exists from pyranges parsing:
            if "Name" in gr.columns:
                gr_subset = gr[["Chromosome", "Start", "End", "Name"]].copy()
            elif (
                len(gr.columns) >= 4 and gr.columns[3] != "Score"
            ):  # Check if 4th column exists and is not score
                # If BED has no header, pyranges might name it like 'df.columns[3]'
                # This is a bit heuristic. The manual parser is more reliable for unheaded/simple BED.
                gr_subset = gr[["Chromosome", "Start", "End", gr.columns[3]]].copy()
                gr_subset.rename(
                    columns={gr.columns[3]: "Name"}, inplace=True
                )  # Standardize to 'Name'
            else:  # Fallback if name column is not clear, create a default '.'
                gr_subset = gr[["Chromosome", "Start", "End"]].copy()
                gr_subset["Name"] = "."

            for _, row in gr_subset.iterrows():
                chrom = row["Chromosome"]
                # Add 'chr' prefix if it doesn't exist (handle common UCSC/Ensembl difference)
                chrom_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
                ranges.append((chrom_str, int(row["Start"]), int(row["End"]), row["Name"]))

            if ranges:
                logging.info(f"Extracted {len(ranges)} ranges from {bed_file_path} using pyranges.")
                return ranges
            else:
                logging.warning(
                    f"Pyranges found no ranges or failed to extract name column from {bed_file_path}. Attempting manual parsing."
                )
        else:
            logging.warning(
                f"Pyranges read an empty BED file: {bed_file_path}. Attempting manual parsing."
            )

    except Exception as e:
        logging.warning(f"Error reading BED file with pyranges: {e}. Attempting manual parsing.")

    # Manual parsing fallback
    ranges = []  # Reset ranges for manual parsing
    try:
        with open(bed_file_path, "r") as f:
            line_number = 0
            for line in f:
                line_number += 1
                line = line.strip()
                if (
                    not line
                    or line.startswith("#")
                    or line.startswith("track")
                    or line.startswith("browser")
                ):
                    continue
                fields = line.split("\t")
                if len(fields) < 3:
                    logging.warning(
                        f"Line {line_number} in {bed_file_path} has fewer than 3 columns, skipping: {line}"
                    )
                    continue

                chrom = fields[0]
                name = (
                    fields[3] if len(fields) >= 4 else f"range_{line_number}"
                )  # Use placeholder if name missing

                try:
                    start = int(fields[1])
                    end = int(fields[2])
                except ValueError:
                    logging.warning(
                        f"Line {line_number} in {bed_file_path} has non-integer coordinates, skipping: {line}"
                    )
                    continue
                if start < 0 or end < 0 or start >= end:
                    logging.warning(
                        f"Line {line_number} in {bed_file_path} has invalid coordinates (start<0, end<0, or start>=end), skipping: {chrom}:{start}-{end}"
                    )
                    continue

                chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                ranges.append((chrom_str, start, end, name))

        if not ranges:
            logging.error(
                f"No valid ranges extracted from BED file after all attempts: {bed_file_path}"
            )
            raise ValueError(f"No valid ranges in BED file: {bed_file_path}")
        else:
            logging.info(
                f"Extracted {len(ranges)} ranges from {bed_file_path} using manual parsing."
            )
    except Exception as e:
        logging.error(f"Error manually reading BED file {bed_file_path}: {e}")
        raise IOError(f"Error reading BED file {bed_file_path}: {e}") from e
    return ranges


def expand_range(chrom, start, end, upstream, downstream, chrom_len=None):
    """
    Expands a genomic range by upstream and downstream amounts.

    Args:
        chrom (str): Chromosome name
        start (int): Original start coordinate (0-based)
        end (int): Original end coordinate (0-based, exclusive)
        upstream (int): Number of bases to extend upstream (toward smaller coordinates)
        downstream (int): Number of bases to extend downstream (toward larger coordinates)
        chrom_len (int, optional): Length of chromosome for bounds checking

    Returns:
        tuple: (expanded_start, expanded_end) with bounds checking applied
    """
    # Apply expansion
    expanded_start = start - upstream  # Subtract from start (upstream = toward smaller coordinates)
    expanded_end = end + downstream  # Add to end (downstream = toward larger coordinates)

    # Apply bounds checking
    expanded_start = max(0, expanded_start)  # Ensure start is not negative

    if chrom_len is not None:
        expanded_end = min(chrom_len, expanded_end)  # Ensure end doesn't exceed chromosome length

    # Ensure start < end after expansion
    if expanded_start >= expanded_end:
        logging.debug(
            f"Range expansion resulted in invalid range for {chrom}:{start}-{end} "
            f"(upstream={upstream}, downstream={downstream}): "
            f"expanded to {expanded_start}-{expanded_end}. Using original range."
        )
        return start, end

    return expanded_start, expanded_end


def calculate_bb_stats(bb_file, chrom, start, end, summary_type="coverage"):
    """
    Calculates summary statistics for a specified region in a BigBed file.
    (This function is for numerical stats, not name extraction)
    """
    if start >= end:
        logging.debug(
            f"Invalid range for BigBed stats: start ({start}) >= end ({end}). Chrom: {chrom}. Returning 0.0"
        )
        return 0.0  # Or np.nan depending on desired behavior for invalid ranges
    if chrom not in bb_file.chroms():
        logging.debug(f"Chromosome '{chrom}' not found in BigBed file for stats. Returning 0.0")
        return 0.0  # Or np.nan

    chrom_len = bb_file.chroms(chrom)
    # Clip query region to chromosome bounds
    query_start = max(0, start)
    query_end = min(chrom_len, end)

    if query_start >= query_end:
        logging.debug(
            f"Query region [{query_start}-{query_end}) is empty or outside chromosome bounds for BigBed stats. Chrom: {chrom}. Returning 0.0"
        )
        return 0.0

    region_len = query_end - query_start  # Length of the actual queried region on chromosome

    try:
        entries = bb_file.entries(chrom, query_start, query_end)
    except (
        RuntimeError
    ) as e:  # pyBigWig can raise RuntimeError if chrom not found, though chroms() check should prevent
        logging.warning(
            f"RuntimeError fetching entries for {chrom}:{query_start}-{query_end} in BigBed: {e}. Returning 0.0"
        )
        return 0.0

    if entries is None:  # No overlapping entries
        return 0.0

    # For coverage, mean, min, max based on BED-like entries (presence/absence or simple depth)
    # This implementation assumes simple presence/absence for coverage,
    # and uses a diff array for depth-based stats if entries were to represent depth (not typical for basic BigBed).
    # For simple BigBed (like TF binding sites), coverage is fraction of bases covered by *any* entry.

    # Calculate exact coverage by merging intervals
    intervals = []
    for entry_s, entry_e, _ in entries:
        # Clip entry to the query region
        clipped_entry_s = max(entry_s, query_start)
        clipped_entry_e = min(entry_e, query_end)
        if clipped_entry_s < clipped_entry_e:
            intervals.append((clipped_entry_s, clipped_entry_e))

    if not intervals:
        return 0.0

    intervals.sort()
    merged_intervals = []
    current_start, current_end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start < current_end:  # Overlap or contiguous
            current_end = max(current_end, next_end)
        else:  # Gap
            merged_intervals.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged_intervals.append((current_start, current_end))

    total_covered_bases_exact = sum(m_end - m_start for m_start, m_end in merged_intervals)

    if summary_type == "coverage":
        if region_len == 0:
            return 0.0
        return total_covered_bases_exact / region_len

    # For other stats (mean, min, max), we need to interpret BigBed entries differently.
    # If BigBed entries have scores that represent value, that's one thing.
    # If it's just presence/absence, 'mean', 'min', 'max' of "depth" (number of overlapping features)
    # The original diff_array approach is for calculating depth profile.

    # Re-implementing depth calculation for mean/min/max of overlap counts
    # This diff_array is relative to the query_start
    diff_array = np.zeros(region_len + 1, dtype=np.int32)
    for entry_s, entry_e, _ in entries:
        # Clip entry to the query region and make relative to query_start
        rel_s = max(entry_s, query_start) - query_start
        rel_e = min(entry_e, query_end) - query_start
        if rel_s < rel_e:  # Ensure valid interval after clipping and making relative
            diff_array[rel_s] += 1
            diff_array[rel_e] -= 1

    depth = np.cumsum(diff_array[:-1])  # Depth at each base within the query region

    if depth.size == 0:  # Should not happen if region_len > 0
        return 0.0

    # For mean/min/max, consider only bases that are covered by at least one feature
    # or consider all bases in the region (where uncovered is depth 0)?
    # Original code: covered_bases_mask = depth > 0; covered_depths = depth[covered_bases_mask]
    # This means stats are on the depth values of *covered* bases only.

    if summary_type == "mean":
        # Mean depth over the *entire query region* (including 0s for uncovered parts)
        # return np.mean(depth)
        # Mean depth over *covered bases only* (if any are covered)
        covered_depths = depth[depth > 0]
        return np.mean(covered_depths) if covered_depths.size > 0 else 0.0
    elif summary_type == "min":
        # Min depth over *covered bases only*
        covered_depths = depth[depth > 0]
        return np.min(covered_depths) if covered_depths.size > 0 else 0.0
    elif summary_type == "max":
        # Max depth over the entire region
        return np.max(depth) if depth.size > 0 else 0.0
    elif summary_type == "sum":  # Sum of depths (total base-pairs covered, accounting for overlaps)
        return np.sum(depth)
    else:
        logging.error(f"Error: Unknown summary type '{summary_type}' for BigBed numerical stats.")
        return np.nan  # Use NaN for unknown stat type


# --- NEW FUNCTION for extracting names from BigBed entries ---
def get_bigbed_entry_names(bb_file, chrom, start, end, name_field_index):
    """
    Extracts unique names from BigBed entries overlapping a given genomic range.

    Args:
        bb_file (pyBigWig file handle): An open BigBed file handle.
        chrom (str): Chromosome name.
        start (int): Start coordinate (0-based).
        end (int): End coordinate (0-based, exclusive).
        name_field_index (int): 0-based index of the name field in the 'rest'
                                string of a BigBed entry (after splitting by tab).

    Returns:
        str: A comma-separated string of unique, sorted names.
        np.nan: If no overlapping entries with names are found, or if an error occurs.
    """
    if start >= end:
        logging.debug(
            f"Invalid range for name extraction: start ({start}) >= end ({end}). Chrom: {chrom}"
        )
        return np.nan

    # Chromosome existence check should be done by caller (process_single_range)
    # Clipping to chrom_len also done by caller via query_end

    try:
        entries = bb_file.entries(chrom, start, end)
    except RuntimeError as e:  # Should be rare if chrom check is done prior
        logging.warning(
            f"RuntimeError fetching entries for name extraction in {chrom}:{start}-{end}: {e}"
        )
        return np.nan

    if entries is None:
        logging.debug(f"No BigBed entries found in {chrom}:{start}-{end} for name extraction.")
        return np.nan

    found_names = set()
    for _, _, rest_string in entries:  # entry_start, entry_end not used here, only overlap matters
        if rest_string:
            fields = rest_string.split("\t")
            if len(fields) > name_field_index:
                name = fields[name_field_index]
                if name and name != ".":  # Ignore empty or placeholder names
                    found_names.add(name)
            # else:
            #     logging.debug(f"Entry in {chrom}:{start}-{end} has too few fields for name_field_index {name_field_index}: {rest_string}")

    if not found_names:
        logging.debug(f"No valid names extracted from BigBed entries in {chrom}:{start}-{end}.")
        return np.nan

    return ",".join(sorted(list(found_names)))


def process_single_range(range_tuple, file_configs, file_manager):
    """
    Worker function executed by threads to process a single genomic range.
    Now supports extracting names from BigBed files and range expansion.
    """
    chrom, start, end, range_name = range_tuple  # Unpack range_name (from input BED)
    range_results = {
        "chromosome": chrom,
        "start": start,
        "end": end,
        "name": range_name,
    }  # Add input range_name
    logging.debug(f"Worker processing range: {chrom}:{start}-{end} (Input Name: {range_name})")

    for config in file_configs:
        file_path = config["path"]
        stat_type = config["stat"]
        col_name = config["name"]  # This is the output column name
        upstream = config.get("upstream", 0)
        downstream = config.get("downstream", 0)
        value = np.nan  # Default to NaN

        bw = file_manager.get_file_handle(file_path)
        if bw is None:
            logging.warning(
                f"Could not access file {file_path} for range {chrom}:{start}-{end}. Setting {col_name} to NaN."
            )
            range_results[col_name] = value
            continue

        try:
            file_chroms = bw.chroms()
            if chrom not in file_chroms:
                logging.debug(
                    f"Chromosome '{chrom}' not found in {file_path} for range {chrom}:{start}-{end}. Setting {col_name} to NaN."
                )
                # value remains np.nan
            else:
                chrom_len = file_chroms[chrom]

                # Apply range expansion if specified
                if upstream > 0 or downstream > 0:
                    expanded_start, expanded_end = expand_range(
                        chrom, start, end, upstream, downstream, chrom_len
                    )
                    logging.debug(
                        f"Expanded range {chrom}:{start}-{end} by upstream={upstream}, downstream={downstream} "
                        f"to {chrom}:{expanded_start}-{expanded_end} for {file_path}"
                    )
                else:
                    expanded_start, expanded_end = start, end

                # Effective query region, clipped to chromosome bounds (additional safety check)
                query_start = max(0, expanded_start)
                query_end = min(expanded_end, chrom_len)

                if query_start >= query_end:  # If range is outside chrom or invalid after clipping
                    expansion_info = (
                        f" (expanded from {start}-{end} by upstream={upstream}, downstream={downstream})"
                        if (upstream > 0 or downstream > 0)
                        else ""
                    )
                    logging.debug(
                        f"Effective query range [{query_start}-{query_end}) is invalid for chromosome '{chrom}' (len {chrom_len}) "
                        f"in {file_path} for input range {chrom}:{start}-{end}{expansion_info}. Setting {col_name} to NaN."
                    )
                    # value remains np.nan
                else:
                    # --- Logic for BigWig files ---
                    if bw.isBigWig():
                        if stat_type == "extract_names":
                            logging.warning(
                                f"'extract_names' stat is not applicable to BigWig files. File: {file_path}. "
                                f"Range: {chrom}:{query_start}-{query_end}. Setting {col_name} to NaN."
                            )
                            # value remains np.nan
                        else:  # Numerical stats for BigWig
                            # pyBigWig stats() expects start < end. query_start < query_end is guaranteed here.
                            result_val_list = bw.stats(
                                chrom, query_start, query_end, type=stat_type, nBins=1
                            )
                            if result_val_list is not None and result_val_list[0] is not None:
                                value = result_val_list[0]
                                logging.debug(
                                    f"BigWig stat '{stat_type}' = {value} for {chrom}:{query_start}-{query_end} from {file_path}"
                                )
                            else:
                                logging.debug(
                                    f"No data for BigWig stat '{stat_type}' in {chrom}:{query_start}-{query_end} "
                                    f"from {file_path}. Setting {col_name} to NaN."
                                )
                                # value remains np.nan
                    # --- Logic for BigBed files ---
                    elif bw.isBigBed():
                        if stat_type == "extract_names":
                            name_field_idx = config.get(
                                "name_field_index", 3
                            )  # Get from config, default 3
                            value = get_bigbed_entry_names(
                                bw, chrom, query_start, query_end, name_field_idx
                            )
                            if value is not np.nan:
                                logging.debug(
                                    f"BigBed extracted names '{value}' for {chrom}:{query_start}-{query_end} from {file_path}"
                                )
                            # else, value is np.nan, message logged by get_bigbed_entry_names
                        else:  # Numerical stats for BigBed
                            value = calculate_bb_stats(bw, chrom, query_start, query_end, stat_type)
                            if value is not np.nan and not (
                                isinstance(value, float) and np.isnan(value)
                            ):  # Check for actual value vs nan
                                logging.debug(
                                    f"BigBed stat '{stat_type}' = {value} for {chrom}:{query_start}-{query_end} from {file_path}"
                                )
                            # else, value might be 0.0 or np.nan, handled by calculate_bb_stats
                    else:
                        logging.error(
                            f"Unsupported file type for {file_path} (neither BigWig nor BigBed). Setting {col_name} to NaN."
                        )
                        # value remains np.nan
        except (
            RuntimeError
        ) as e:  # Catch pyBigWig runtime errors (e.g. file corruption, specific access issues)
            logging.error(
                f"Runtime error processing file {file_path} for range {chrom}:{start}-{end}: {e}. Setting {col_name} to NaN."
            )
            # value remains np.nan
        except Exception as e:
            logging.error(
                f"Unexpected error processing file {file_path} for range {chrom}:{start}-{end}: {e}. Setting {col_name} to NaN.",
                exc_info=True,  # Log traceback for unexpected errors
            )
            # value remains np.nan

        range_results[col_name] = value
    return range_results


def process_range_batch(range_batch, file_configs, file_manager):
    """
    Process a batch of ranges to reduce threading overhead for small ranges.
    """
    batch_results = []
    for range_tuple in range_batch:
        result = process_single_range(range_tuple, file_configs, file_manager)
        if result:
            batch_results.append(result)
    return batch_results


def process_range_batch_with_local_manager(
    range_batch,
    file_configs,
    url_timeout=30,
    max_retries=3,
    verify_ssl=True,
    keep_downloaded_files=False,
    base_storage_directory=None,
):
    """
    Process a batch of ranges with a local file manager for process-based parallelization.
    """
    # Create a local file manager for this process
    local_file_manager = BigWigFileManager(
        url_timeout=url_timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        keep_downloaded_files=keep_downloaded_files,
        base_storage_directory=base_storage_directory,
    )

    try:
        return process_range_batch(range_batch, file_configs, local_file_manager)
    finally:
        # Clean up local file manager
        local_file_manager.close_all()


def process_single_range_with_local_manager(
    range_tuple,
    file_configs,
    url_timeout=30,
    max_retries=3,
    verify_ssl=True,
    keep_downloaded_files=False,
    base_storage_directory=None,
):
    """
    Worker function for process-based parallelization.
    Each process creates its own file manager to avoid sharing issues.
    """
    # Create a local file manager for this process
    local_file_manager = BigWigFileManager(
        url_timeout=url_timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        keep_downloaded_files=keep_downloaded_files,
        base_storage_directory=base_storage_directory,
    )

    try:
        return process_single_range(range_tuple, file_configs, local_file_manager)
    finally:
        # Clean up local file manager
        local_file_manager.close_all()


def query_bigwig_files(
    ranges,
    file_configs,
    max_workers=None,
    return_type="dataframe",
    url_timeout=30,
    max_retries=3,
    verify_ssl=True,
    preload_files=True,
    keep_downloaded_files=False,
    clear_cache_on_startup=False,
    base_storage_directory=None,
    use_processes=True,
    batch_size=None,
    show_progress=None,
    quiet=False,
):
    """
    Queries multiple BigWig/BigBed files over specified genomic ranges using parallel processing.

    Args:
        use_processes (bool): If True, use ProcessPoolExecutor for better parallelization.
                             If False, use ThreadPoolExecutor (limited by Python's GIL).
                             Defaults to True for optimal performance.
        batch_size (int): Number of ranges to process in each batch. If None, automatically
                         calculated based on the number of ranges and workers.
                         Batching reduces threading overhead for many small ranges.
    """
    if not ranges:
        logging.warning("Input 'ranges' list is empty. Returning empty result.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])
    if not file_configs:
        logging.warning("Input 'file_configs' list is empty. Returning empty result.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    results_list = []  # Renamed from 'results' to avoid conflict if used as a variable name
    file_manager = BigWigFileManager(
        url_timeout=url_timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        keep_downloaded_files=keep_downloaded_files,
        base_storage_directory=base_storage_directory,
    )

    if clear_cache_on_startup:
        logging.info("Clearing persistent cache as requested...")
        file_manager.clear_persistent_cache()

    try:
        if max_workers is None:
            max_workers = os.cpu_count() or 1  # Ensure at least 1 worker
            logging.info(f"Using default max_workers: {max_workers}")
        else:
            max_workers = max(1, max_workers)  # Ensure at least 1 worker
            logging.info(f"Using specified max_workers: {max_workers}")

        if preload_files:
            unique_paths = sorted(
                list(set(config["path"] for config in file_configs))
            )  # Sorted for consistent logging
            logging.info(f"Preloading {len(unique_paths)} unique file paths...")
            
            # Create progress manager and bar for file preloading
            progress_manager = get_progress_manager(show_progress=show_progress, quiet=quiet)
            preload_progress = progress_manager.create_bar(
                total=len(unique_paths), 
                desc="Downloading/caching files"
            )
            
            try:
                file_manager.preload_remote_files(unique_paths, progress_bar=preload_progress)
            finally:
                preload_progress.close()

        # Choose executor type based on workload characteristics
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        executor_name = "ProcessPoolExecutor" if use_processes else "ThreadPoolExecutor"

        # Calculate optimal batch size for small ranges
        if batch_size is None:
            # Auto-calculate batch size: aim for at least 2x more batches than workers
            # but don't make batches too small (min 1) or too large (max 50)
            total_ranges = len(ranges)
            optimal_batches = max_workers * 2
            auto_batch_size = max(1, min(50, total_ranges // optimal_batches))
            batch_size = auto_batch_size if total_ranges > max_workers * 10 else 1

        # Optimize worker count for I/O vs CPU bound workloads
        if not use_processes and max_workers > os.cpu_count() * 2:
            # For I/O-bound work (threads), we can use more workers than CPU cores
            logging.info(f"I/O-bound workload detected, keeping max_workers at {max_workers}")
        elif use_processes and max_workers > os.cpu_count():
            # For CPU-bound work (processes), limit to CPU cores
            original_workers = max_workers
            max_workers = os.cpu_count()
            logging.info(
                f"CPU-bound workload: reduced max_workers from {original_workers} to {max_workers}"
            )

        logging.info(f"Using {executor_name} with {max_workers} workers, batch_size={batch_size}")

        # Create batches if batch_size > 1
        if batch_size > 1:
            range_batches = [ranges[i : i + batch_size] for i in range(0, len(ranges), batch_size)]
            logging.info(f"Created {len(range_batches)} batches from {len(ranges)} ranges")
        else:
            range_batches = [[r] for r in ranges]  # Each range is its own batch

        with executor_class(max_workers=max_workers) as executor:
            if use_processes:
                # For processes, we need to pass file configurations and manager settings
                # since file handles can't be shared across processes
                if batch_size > 1:
                    worker_func = partial(
                        process_range_batch_with_local_manager,
                        file_configs=file_configs,
                        url_timeout=url_timeout,
                        max_retries=max_retries,
                        verify_ssl=verify_ssl,
                        keep_downloaded_files=keep_downloaded_files,
                        base_storage_directory=base_storage_directory,
                    )
                else:
                    worker_func = partial(
                        process_single_range_with_local_manager,
                        file_configs=file_configs,
                        url_timeout=url_timeout,
                        max_retries=max_retries,
                        verify_ssl=verify_ssl,
                        keep_downloaded_files=keep_downloaded_files,
                        base_storage_directory=base_storage_directory,
                    )
            else:
                # For threads, use the shared file manager
                if batch_size > 1:
                    worker_func = partial(
                        process_range_batch,
                        file_configs=file_configs,
                        file_manager=file_manager,
                    )
                else:
                    worker_func = partial(
                        process_single_range,
                        file_configs=file_configs,
                        file_manager=file_manager,
                    )

            # Submit work (either individual ranges or batches)
            if batch_size > 1:
                futures = [executor.submit(worker_func, batch) for batch in range_batches]
                logging.info(f"Submitted {len(futures)} batch queries to {max_workers} workers.")
                progress_desc = f"Processing {len(ranges)} ranges in {len(futures)} batches"
                progress_total = len(futures)
            else:
                futures = [
                    executor.submit(worker_func, r[0]) for r in range_batches
                ]  # r[0] since each batch has 1 element
                logging.info(f"Submitted {len(futures)} range queries to {max_workers} workers.")
                progress_desc = f"Processing {len(ranges)} ranges"
                progress_total = len(ranges)

            # Create progress bar for processing
            progress_manager = get_progress_manager(show_progress=show_progress, quiet=quiet)
            process_progress = progress_manager.create_bar(
                total=progress_total,
                desc=progress_desc
            )
            
            try:
                for i, future in enumerate(as_completed(futures)):
                    try:
                        result = future.result()
                        if result:
                            if batch_size > 1 and isinstance(result, list):
                                # Result is a list of dictionaries from batch processing
                                results_list.extend(result)
                            elif isinstance(result, dict):
                                # Single result dictionary
                                results_list.append(result)
                        
                        # Update progress bar
                        if batch_size > 1:
                            # For batches, we estimate progress based on batch completion
                            process_progress.update(1)
                        else:
                            process_progress.update(1)
                            
                    except Exception as e:
                        logging.error(f"Error retrieving result for a range query: {e}", exc_info=True)
                        # Still update progress even on error
                        process_progress.update(1)
            finally:
                process_progress.close()
    finally:
        file_manager.close_all()

    if not results_list:
        logging.warning(
            "No results were generated. Check logs for issues with ranges, files, or configurations."
        )
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    logging.info(f"Collected {len(results_list)} results. Creating {return_type}.")

    # Define column order: standard BED-like columns first, then stat/extraction columns
    # The 'name' column here is from the input BED, identifying the range.
    # Stat columns are derived from file_configs.
    stat_col_names = [config["name"] for config in file_configs]
    # Ensure 'name' (from input BED) is distinct from stat column names if there's a clash
    # Though 'name' in range_results is fixed.
    column_order = ["chromosome", "start", "end", "name"] + sorted(
        list(set(stat_col_names))
    )  # Sort stat cols for consistency

    df = pd.DataFrame(results_list)

    # change column name from 'name' to 'transcript_id'
    if "name" in df.columns:
        df.rename(columns={"name": "transcript_id"}, inplace=True)

    # Reorder columns to ensure standard ones are first, followed by dynamic stat columns
    # Some stat_col_names might be missing if all values were NaN and thus column not created by DataFrame constructor.
    # So, filter column_order to existing columns in df.
    final_column_order = [col for col in column_order if col in df.columns]
    # Add any other columns that might have been created but not in explicit order (should not happen with current logic)
    for col in df.columns:
        if col not in final_column_order:
            final_column_order.append(col)

    df = df[final_column_order]

    if return_type.lower() == "dataframe":
        return df
    elif return_type.lower() == "array":
        return df.to_numpy()
    else:
        logging.warning(f"Unknown return_type '{return_type}'. Returning DataFrame.")
        return df


def save_results(df, output_path, output_format=None):
    """
    Saves a results DataFrame to a specified file path and format.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:  # Check if df is a DataFrame
        logging.error("Cannot save empty or invalid (non-DataFrame) results.")
        return False

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Created output directory: {output_dir}")

        if output_format is None:
            _, ext = os.path.splitext(output_path.lower())
            if ext in (".csv", ".txt", ".tsv"):  # Added tsv
                output_format = "csv"
            elif ext in (".parquet", ".pq"):
                output_format = "parquet"
            else:
                logging.warning(
                    f"Unable to detect format from extension '{ext}'. Defaulting to CSV."
                )
                output_format = "csv"

        if output_format.lower() == "csv":
            sep = "\t" if output_path.lower().endswith(".tsv") else ","
            df.to_csv(output_path, index=False, sep=sep)
            logging.info(f"Results saved as CSV (separator: '{sep}') to {output_path}")
        elif output_format.lower() == "parquet":
            try:
                df.to_parquet(output_path, index=False)
                logging.info(f"Results saved as Parquet to {output_path}")
            except ImportError:
                logging.error(
                    "Failed to save as Parquet. 'pyarrow' or 'fastparquet' library is required. Please install it."
                )
                return False
        else:
            logging.error(f"Unsupported output format: {output_format}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}", exc_info=True)
        return False


def process_bigwig_query(
    bed_file: str,
    config_file: str,
    output_file: Optional[str] = None,
    threads: Optional[int] = None,
    log_level: str = "INFO",
    return_type: str = "dataframe",
    output_format: Optional[str] = None,
    url_timeout: int = 30,
    max_retries: int = 3,
    verify_ssl: bool = True,
    preload_files: bool = True,
    keep_downloaded_files: bool = False,
    clear_cache_on_startup: bool = False,
    base_storage_directory: Optional[str] = None,
    use_processes: bool = True,
    batch_size: Optional[int] = None,
    show_progress: bool = None,
    quiet: bool = False,
) -> Union[pd.DataFrame, None]:
    """
    Process BigWig/BigBed files to extract statistics for genomic ranges.

    This function coordinates the entire BigWig/BigBed query process, from loading
    configuration and genomic ranges to querying remote/local files and returning
    results. It supports caching, multiprocessing, and various output formats.

    Args:
        bed_file: Path to BED file containing genomic ranges (chrom, start, end, name)
        config_file: Path to YAML configuration file specifying BigWig/BigBed files and statistics
        output_file: Optional path to save results. If None, results are only returned
        threads: Number of worker threads/processes. Defaults to CPU count
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        return_type: Format of returned data ('dataframe' or 'array')
        output_format: Output file format (inferred from extension if None)
        url_timeout: Timeout in seconds for downloading remote files
        max_retries: Maximum retry attempts for failed downloads
        verify_ssl: Whether to verify SSL certificates for HTTPS downloads
        preload_files: Whether to preload remote files before processing
        keep_downloaded_files: Whether to keep downloaded files in persistent cache
        clear_cache_on_startup: Whether to clear persistent cache before processing
        base_storage_directory: Base directory for file caching
        use_processes: Whether to use multiprocessing (True) or multithreading (False)
        batch_size: Number of ranges to process per batch

    Returns:
        pandas.DataFrame or numpy.ndarray containing query results, or None on error

    Example:
        >>> import pandas as pd
        >>> from features.bwq import process_bigwig_query
        >>>
        >>> # Process BigWig files for genomic ranges
        >>> results = process_bigwig_query(
        ...     bed_file="ranges.bed",
        ...     config_file="tracks.yaml",
        ...     output_file="results.csv",
        ...     threads=4
        ... )
        >>> print(results.head())
    """
    # Setup logging (idempotent if already set, or sets if called as library)
    # setup_logging(log_level) # This might be called multiple times if main() also calls it.
    # Better to ensure it's called once at the top level.

    logging.info(f"Loading configuration from {config_file}")
    file_configs = load_config_from_yaml(config_file)
    if not file_configs:  # If config loading results in no valid tasks
        logging.error("No valid configurations loaded. Aborting.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    logging.info(f"Reading genomic ranges from {bed_file}")
    ranges = read_ranges_from_bed(bed_file)
    if not ranges:
        logging.error(f"No valid ranges found in BED file: {bed_file}. Aborting.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    logging.info(f"Processing queries with {threads or os.cpu_count() or 1} workers")
    results_df = query_bigwig_files(  # Ensure results are DataFrame for saving
        ranges,
        file_configs,
        max_workers=threads,
        return_type="dataframe",  # Always get DataFrame for potential saving
        url_timeout=url_timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        preload_files=preload_files,
        keep_downloaded_files=keep_downloaded_files,
        clear_cache_on_startup=clear_cache_on_startup,
        base_storage_directory=base_storage_directory,
        use_processes=use_processes,
        batch_size=batch_size,
        show_progress=show_progress,
        quiet=quiet,
    )

    if output_file and not results_df.empty:
        save_results(results_df, output_file, output_format)
    elif output_file and results_df.empty:
        logging.warning(
            f"Results DataFrame is empty. Output file {output_file} will not be created."
        )

    # Return in the originally requested format if different from DataFrame
    if return_type.lower() == "array":
        return results_df.to_numpy() if isinstance(results_df, pd.DataFrame) else np.array([])
    return results_df


def query_bigwig_ranges(
    bed_file: str, config_file: str, output_file: Optional[str] = None, 
    show_progress: bool = True, **kwargs
) -> pd.DataFrame:
    """
    Simple API entry point for querying BigWig/BigBed files.

    This is a simplified interface to process_bigwig_query with sensible defaults
    for programmatic usage. All additional parameters can be passed via kwargs.

    Args:
        bed_file: Path to BED file containing genomic ranges
        config_file: Path to YAML configuration file
        output_file: Optional output file path
        **kwargs: Additional parameters passed to process_bigwig_query

    Returns:
        pandas.DataFrame: Query results

    Raises:
        FileNotFoundError: If input files don't exist
        ValueError: If configuration is invalid
        RuntimeError: If processing fails

    Example:
        >>> from features.bwq import query_bigwig_ranges
        >>>
        >>> # Basic usage
        >>> df = query_bigwig_ranges("genes.bed", "tracks.yaml")
        >>>
        >>> # With custom parameters
        >>> df = query_bigwig_ranges(
        ...     "genes.bed",
        ...     "tracks.yaml",
        ...     threads=8,
        ...     keep_downloaded_files=True
        ... )
    """
    # Set logging to INFO level if not specified
    kwargs.setdefault("log_level", "INFO")

    try:
        setup_logging(kwargs.get("log_level", "INFO"))
        result = process_bigwig_query(
            bed_file=bed_file, config_file=config_file, output_file=output_file, 
            show_progress=show_progress, **kwargs
        )

        if result is None:
            raise RuntimeError("Processing failed to return results")

        return result

    except Exception as e:
        logging.error(f"BigWig query failed: {e}")
        raise


def main() -> int:
    """
    Main function to handle command-line arguments and execute the BigWig query process.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Extract statistics and names from BigWig/BigBed files for genomic regions specified in a BED file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bed", "-b", required=True, help="Path to BED file (chrom, start, end, [name], ...)"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="Path or URL to YAML configuration file"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output file (e.g., .csv, .tsv, .parquet)"
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to CPU cores.",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "tsv", "parquet"],
        help="Output format (overrides extension detection)",
    )
    parser.add_argument(
        "--url-timeout", type=int, default=30, help="Timeout (seconds) for URL connections"
    )
    parser.add_argument(
        "--url-retries", type=int, default=3, help="Max retries for URL connections"
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for HTTPS URLs",
    )
    parser.add_argument(
        "--disable-preload", action="store_true", help="Disable preloading of remote files"
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded remote files in a persistent cache",
    )
    parser.add_argument(
        "--clear-cache", action="store_true", help="Clear persistent cache before starting"
    )
    parser.add_argument(
        "--base-storage-dir",
        type=str,
        default=None,
        help="Custom base directory for cache/temp files",
    )
    parser.add_argument(
        "--use-processes",
        action="store_true",
        default=True,
        help="Use ProcessPoolExecutor for better parallelization (default: True)",
    )
    parser.add_argument(
        "--use-threads",
        action="store_true",
        help="Use ThreadPoolExecutor instead of ProcessPoolExecutor (overrides --use-processes)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of ranges to process per batch (auto-calculated if not specified)",
    )

    # Add progress-related arguments
    update_cli_args(parser)

    args = parser.parse_args()

    setup_logging(args.log_level)  # Setup logging once here

    # Determine execution mode: processes by default, threads if explicitly requested
    use_processes = args.use_processes and not args.use_threads

    # Resolve progress settings from CLI arguments
    progress_settings = resolve_progress_settings(args)

    try:
        results = process_bigwig_query(
            bed_file=args.bed,
            config_file=args.config,
            output_file=args.output,
            threads=args.threads,
            log_level=args.log_level,  # Passed but logging already set up
            output_format=args.format,
            url_timeout=args.url_timeout,
            max_retries=args.url_retries,
            verify_ssl=not args.no_verify_ssl,
            preload_files=not args.disable_preload,
            keep_downloaded_files=args.keep_downloads,
            clear_cache_on_startup=args.clear_cache,
            base_storage_directory=args.base_storage_dir,
            use_processes=use_processes,
            batch_size=args.batch_size,
            **progress_settings,  # Pass progress settings
        )

        if isinstance(results, pd.DataFrame) and results.empty:
            logging.warning("Processing finished, but the resulting DataFrame is empty.")
            # If output was specified, an empty file might not be written, or an empty one is.
            # process_bigwig_query handles logging for this.
            # Exit with non-zero status if no results and output was expected.
            if args.output:
                return 1
        elif isinstance(results, np.ndarray) and results.size == 0:
            logging.warning("Processing finished, but the resulting array is empty.")
            if args.output:
                return 1
        elif results is None:  # Should ideally not happen if errors are caught
            logging.error("Processing failed to return results.")
            return 1

        logging.info("Processing completed successfully.")
        return 0

    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
        return 1
    except ValueError as e:  # For config errors, etc.
        logging.error(f"Configuration or parameter error: {e}")
        return 1
    except IOError as e:  # For file read/write issues not covered by FileNotFoundError
        logging.error(f"File reading/writing error: {e}")
        return 1
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
