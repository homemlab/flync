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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from functools import partial

import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import yaml


# --- File Manager Class ---
class BigWigFileManager:
    """
    Manages BigWig/BigBed file handles for efficient access, including local caching of remote files.

    This class handles opening local and remote files, downloading remote files
    with retries, caching downloaded files (persistently or transiently),
    and managing file handles to avoid repeatedly opening the same file.
    It uses an ExitStack to ensure proper cleanup of resources.
    """

    def __init__(self, url_timeout=30, max_retries=3, base_storage_directory=None, verify_ssl=True, keep_downloaded_files=False):
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
            "User-Agent": "BigWigQuery/1.0", # Consider making this more informative if needed
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
                            logging.warning(f"Could not remove partially downloaded file {target_local_path}: {unlink_e}")
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
            # For URLs, we need to handle downloading with thread safety
            with self.download_lock:
                if file_path in self.file_handles:
                    return self.file_handles[file_path]
                if file_path in self.downloading:
                    logging.debug(f"Waiting for {file_path} to be downloaded by another thread")
                    # Fall through to wait logic outside this lock

            # Wait for the file to be downloaded by another thread (if self.downloading was true)
            if file_path in self.downloading: # Check again, as it might have been set above
                max_attempts = 20 # Increased attempts for waiting
                for attempt in range(max_attempts):
                    time.sleep(0.5)  # Short wait
                    with self.handle_lock:
                        if file_path in self.file_handles:
                            return self.file_handles[file_path]
                    if attempt == max_attempts - 1:
                        logging.warning(f"Timeout waiting for {file_path} download from another thread")
                        # Proceed to attempt download by this thread if timeout occurs

            # Re-check and start downloading if necessary
            with self.download_lock:
                if file_path in self.file_handles: # Check again, might have been opened while waiting
                    return self.file_handles[file_path]
                if file_path in self.downloading: # Another thread started download while we waited
                    logging.debug(f"Waiting again for {file_path} download (another thread took over)")
                    # This return will likely trigger retry or failure upstream if the other thread also fails.
                    # Or, implement a more robust wait here. For now, let the outer logic handle.
                    return None # Or loop again with sleep

                # Mark this URL as being downloaded by THIS thread
                self.downloading.add(file_path)

            actual_file_to_open = None
            try:
                if self.keep_downloaded_files:
                    persistent_path = self._get_persistent_cache_path(file_path)
                    if os.path.exists(persistent_path):
                        logging.info(f"Using existing persistent cached file: {persistent_path} for {file_path}")
                        actual_file_to_open = persistent_path
                    else:
                        logging.info(f"Downloading {file_path} to persistent cache: {persistent_path}")
                        actual_file_to_open = self._perform_download(file_path, persistent_path)
                else: # Not keeping files, download to a true temporary file
                    temp_fd, temp_path_for_download = tempfile.mkstemp(
                        suffix=os.path.splitext(file_path)[1], dir=self.transient_temp_dir
                    )
                    os.close(temp_fd) # We only need the path for _perform_download

                    logging.info(f"Downloading {file_path} to transient temp file: {temp_path_for_download}")
                    actual_file_to_open = self._perform_download(file_path, temp_path_for_download)
                    if actual_file_to_open:
                        self.transient_files_to_delete.append(actual_file_to_open) # Mark for deletion

                if actual_file_to_open is None:
                    logging.error(f"Failed to obtain local file for {file_path}")
                    # Ensure cleanup of downloading set
                    with self.download_lock:
                        if file_path in self.downloading:
                            self.downloading.remove(file_path)
                    return None

                # Open the local file (cached or temporary)
                bw = self.exit_stack.enter_context(pyBigWig.open(actual_file_to_open))

                with self.handle_lock:
                    self.file_handles[file_path] = bw
                
                with self.download_lock:
                    self.downloading.remove(file_path)

                logging.info(f"Successfully opened remote BigWig/BigBed file: {file_path} (from {actual_file_to_open})")
                return bw
            except Exception as e:
                logging.error(f"Error opening or processing remote file {file_path}: {e}")
                with self.download_lock:
                    if file_path in self.downloading:
                        self.downloading.remove(file_path)
                # If a transient file was created but opening failed, it should still be in transient_files_to_delete
                return None
        else:
            # Regular local file - simpler handling
            try:
                bw = self.exit_stack.enter_context(pyBigWig.open(file_path))

                # Store the handle for reuse
                with self.handle_lock:
                    self.file_handles[file_path] = bw

                return bw
            except Exception as e:
                logging.error(f"Error opening file {file_path}: {e}")
                return None

    def preload_remote_files(self, file_paths):
        """
        Attempts to download and open handles for all unique remote URLs in the given list.

        This is useful to ensure remote files are available and cached before starting
        parallel processing that might request them concurrently.

        Args:
            file_paths (list[str]): A list of file paths or URLs. Only URLs will be preloaded.

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
        for url in remote_urls:
            if self.get_file_handle(url) is not None:
                success_count += 1

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
            logging.info(f"Persistent cache directory {self.persistent_cache_dir} does not exist. Nothing to clear.")
            return

        logging.info(f"Clearing persistent cache directory: {self.persistent_cache_dir}")
        cleared_count = 0
        error_count = 0
        for filename in os.listdir(self.persistent_cache_dir):
            file_path = os.path.join(self.persistent_cache_dir, filename)
            try:
                # Ensure we only delete files prefixed by our cache marker, if desired
                if filename.startswith("bwq_cache_"):
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        cleared_count += 1
                    elif os.path.isdir(file_path): # Should not happen with current naming
                        shutil.rmtree(file_path)
                        cleared_count +=1
                else:
                    logging.debug(f"Skipping non-cache file in cache directory: {filename}")
            except Exception as e:
                logging.warning(f"Failed to delete {file_path} from cache: {e}")
                error_count += 1
        if cleared_count > 0 or error_count > 0:
            logging.info(f"Cleared {cleared_count} items from persistent cache. Encountered {error_count} errors.")
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
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
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
    """Loads file configurations from a YAML file with support for multiple stats."""
    if not is_url(yaml_path) and not os.path.exists(yaml_path):
        logging.error(f"Configuration file not found: {yaml_path}")
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration file {yaml_path}: {e}")
        raise ValueError(
            f"Error parsing YAML configuration file {yaml_path}: {e}"
        ) from e
    except Exception as e:
        logging.error(f"Error reading configuration file {yaml_path}: {e}")
        raise IOError(f"Error reading configuration file {yaml_path}: {e}") from e

    # Basic validation of the loaded structure
    if not isinstance(config_data, list):
        logging.error(
            f"YAML config should be a list of file configurations, but got {type(config_data)}"
        )
        raise ValueError("YAML config must be a list.")

    expanded_configs = []
    valid_stats = ["mean", "max", "min", "coverage", "std", "sum"]

    for i, item in enumerate(config_data):
        if not isinstance(item, dict):
            logging.error(f"Item #{i+1} in config is not a dictionary. Config: {item}")
            raise ValueError(f"Item #{i+1} in config must be a dictionary.")

        # Check if path exists or is a valid URL
        if "path" not in item or not isinstance(item["path"], str) or not item["path"]:
            logging.error(
                f"Item #{i+1} in config has invalid or missing 'path'. Config: {item}"
            )
            raise ValueError(f"Item #{i+1} in config has invalid or missing 'path'.")

        file_path = item["path"]
        if not is_url(file_path) and not os.path.exists(file_path):
            logging.error(
                f"File specified in config item #{i+1} not found: {file_path}"
            )
            raise FileNotFoundError(
                f"File specified in config item #{i+1} not found: {file_path}"
            )

        # Support both old format (single stat) and new format (multiple stats)
        if "stat" in item and "name" in item:
            # Old format with single stat
            if item["stat"] not in valid_stats:
                logging.error(
                    f"Item #{i+1} in config has unsupported statistic '{item['stat']}'. "
                    f"Supported stats are: {', '.join(valid_stats)}. Config: {item}"
                )
                raise ValueError(
                    f"Item #{i+1} in config has unsupported statistic '{item['stat']}'."
                )

            expanded_configs.append(
                {"path": file_path, "stat": item["stat"], "name": item["name"]}
            )
        elif "stats" in item:
            # New format with multiple stats
            if not isinstance(item["stats"], list):
                logging.error(f"Item #{i+1} 'stats' must be a list. Config: {item}")
                raise ValueError(f"Item #{i+1} 'stats' must be a list.")

            for j, stat_item in enumerate(item["stats"]):
                if not isinstance(stat_item, dict):
                    logging.error(
                        f"Stat item #{j+1} in item #{i+1} is not a dictionary. Config: {stat_item}"
                    )
                    continue

                if "stat" not in stat_item or "name" not in stat_item:
                    logging.error(
                        f"Stat item #{j+1} in item #{i+1} missing required keys. Config: {stat_item}"
                    )
                    continue

                if stat_item["stat"] not in valid_stats:
                    logging.error(
                        f"Stat item #{j+1} in item #{i+1} has unsupported statistic '{stat_item['stat']}'. "
                        f"Supported stats are: {', '.join(valid_stats)}."
                    )
                    continue

                expanded_configs.append(
                    {
                        "path": file_path,
                        "stat": stat_item["stat"],
                        "name": stat_item["name"],
                    }
                )
        else:
            logging.error(
                f"Item #{i+1} must have either 'stat'+'name' or 'stats'. Config: {item}"
            )
            raise ValueError(f"Item #{i+1} must have either 'stat'+'name' or 'stats'.")

    logging.info(
        f"Loaded {len(expanded_configs)} valid file configurations from {yaml_path}"
    )
    return expanded_configs


def read_ranges_from_bed(bed_file_path):
    """Reads genomic ranges (chrom, start, end, name) from a BED file."""
    ranges = []
    if not os.path.exists(bed_file_path):
        logging.error(f"BED file not found: {bed_file_path}")
        raise FileNotFoundError(f"BED file not found: {bed_file_path}")

    # First try using pyranges
    try:
        gr = pr.read_bed(bed_file_path)

        if len(gr) == 0:
            logging.warning(f"No valid ranges found in BED file: {bed_file_path}")
            # Fall back to manual parsing
            raise ValueError("Empty BED file or parsing failed")
    except Exception as e:
        logging.warning(
            f"Error reading BED file with pyranges: {e}. Attempting manual parsing."
        )
        pass

    # If pyranges fails or returns no ranges, try manual parsing
    try:
        with open(bed_file_path, "r") as f:
            line_number = 0
            for line in f:
                line_number += 1
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Skip comment or track lines
                if (
                    line.startswith("#")
                    or line.startswith("track")
                    or line.startswith("browser")
                ):
                    continue

                fields = line.split("\t")

                # Check if we have at least 3 fields (chrom, start, end)
                if len(fields) < 3:
                    logging.warning(
                        f"Line {line_number} has fewer than 3 columns, skipping: {line}"
                    )
                    continue

                chrom = fields[0]
                name = fields[3] if len(fields) >= 4 else "." # BED name field, or "." if missing

                # Try to parse start and end coordinates
                try:
                    start = int(fields[1])
                    end = int(fields[2])
                except ValueError:
                    logging.warning(
                        f"Line {line_number} has non-integer coordinates, skipping: {line}"
                    )
                    continue

                # Validate coordinates
                if start < 0 or end < 0:
                    logging.warning(
                        f"Line {line_number} has negative coordinates, skipping: {chrom}:{start}-{end}"
                    )
                    continue

                if start >= end:
                    logging.warning(
                        f"Line {line_number} has start >= end, skipping: {chrom}:{start}-{end}"
                    )
                    continue

                # Add 'chr' prefix if it doesn't exist
                # This is common in UCSC dm6 track files
                chrom_str = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                ranges.append((chrom_str, start, end, name))

        if not ranges:
            logging.warning(f"No valid ranges extracted from BED file: {bed_file_path}")
        else:
            logging.info(
                f"Extracted {len(ranges)} ranges from {bed_file_path} using manual parsing"
            )

    except Exception as e:
        logging.error(f"Error manually reading BED file {bed_file_path}: {e}")
        raise IOError(f"Error reading BED file {bed_file_path}: {e}") from e

    return ranges


def calculate_bb_stats(bb_file, chrom, start, end, summary_type="coverage"):
    """
    Calculates summary statistics for a specified region in a BigBed file.

    Handles calculation of coverage (fraction of bases covered), mean depth,
    min depth, and max depth within the region.

    Args:
        bb_file (pyBigWig file handle): An open BigBed file handle.
        chrom (str): Chromosome name.
        start (int): Start coordinate (0-based).
        end (int): End coordinate (0-based, exclusive).
        summary_type (str): The type of summary statistic to calculate.
                            Options: 'coverage', 'mean', 'min', 'max'. Defaults to 'coverage'.

    Returns:
        float: The calculated statistic value.
        None: If the chromosome is not found or an unknown summary type is requested.
        0.0: If the region is empty, invalid, or has no overlapping entries.
    """
    if start >= end:
        logging.error(
            f"Error: Start position ({start}) must be less than end position ({end})."
        )
        return None
    if chrom not in bb_file.chroms():
        logging.error(f"Error: Chromosome '{chrom}' not found in the BigBed file.")
        return None

    chrom_len = bb_file.chroms(chrom)
    if start < 0:
        start = 0
    if end > chrom_len:
        end = chrom_len
    if start >= end:
        logging.warning(
            f"Warning: Query region [{start}-{end}) is empty or outside chromosome bounds ({chrom_len})."
        )
        if summary_type == "coverage":
            return 0.0
        else:
            return 0.0

    region_len = end - start
    entries = bb_file.entries(chrom, start, end)

    if entries is None:
        return 0.0

    diff_array = np.zeros(region_len + 1, dtype=np.int32)
    total_covered_bases_exact = 0
    intervals = []

    for entry_start, entry_end, _ in entries:
        clipped_start = max(entry_start, start)
        clipped_end = min(entry_end, end)

        if clipped_start < clipped_end:
            intervals.append((clipped_start, clipped_end))
            diff_array[clipped_start - start] += 1
            diff_array[clipped_end - start] -= 1

    if not intervals:
        return 0.0

    intervals.sort()
    merged_intervals = []
    if intervals:
        current_start, current_end = intervals[0]
        for next_start, next_end in intervals[1:]:
            if next_start < current_end:
                current_end = max(current_end, next_end)
            else:
                merged_intervals.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged_intervals.append((current_start, current_end))

    for m_start, m_end in merged_intervals:
        total_covered_bases_exact += m_end - m_start

    if summary_type == "coverage":
        if region_len == 0:
            return 0.0
        return total_covered_bases_exact / region_len

    depth = np.cumsum(diff_array[:-1])
    covered_bases_mask = depth > 0
    covered_depths = depth[covered_bases_mask]

    if covered_depths.size == 0:
        return 0.0

    if summary_type == "mean":
        return np.mean(covered_depths)
    elif summary_type == "min":
        return np.min(covered_depths)
    elif summary_type == "max":
        return np.max(depth)
    else:
        logging.error(f"Error: Unknown summary type '{summary_type}'")
        return None


def process_single_range(range_tuple, file_configs, file_manager):
    """
    Worker function executed by threads to process a single genomic range.

    Fetches the required file handle from the shared manager, queries the
    BigWig/BigBed file for the specified range and statistics defined in
    `file_configs`, and returns the results for that range.

    Args:
        range_tuple (tuple): A tuple containing (chromosome, start, end, name).
                             'name' is the identifier for the range (e.g., from BED column 4).
        file_configs (list[dict]): List of file configuration dictionaries, where each
                                   dictionary specifies 'path', 'stat', and 'name'.
        file_manager (BigWigFileManager): The shared manager instance for file handles.

    Returns:
        dict: A dictionary containing the range coordinates ('chromosome', 'start', 'end', 'name')
              and the calculated statistics as key-value pairs (e.g., {'column_name': value}).
              Values will be np.nan if a file/chromosome is inaccessible or contains no data
              for the range/statistic.
    """
    chrom, start, end, name = range_tuple # Unpack name here
    range_results = {"chromosome": chrom, "start": start, "end": end, "name": name} # Add name to results
    logging.debug(f"Worker processing range: {chrom}:{start}-{end} (Name: {name})")

    for config in file_configs:
        file_path = config["path"]
        stat_type = config["stat"]
        col_name = config["name"]
        value = np.nan

        bw = file_manager.get_file_handle(file_path)
        if bw is None:
            logging.warning(
                f"Could not access file {file_path} for range {chrom}:{start}-{end}"
            )
            range_results[col_name] = value
            continue

        try:
            file_chroms = bw.chroms()
            if chrom not in file_chroms:
                logging.debug(
                    f"Chromosome '{chrom}' not found in {file_path} for range {chrom}:{start}-{end}. Setting {col_name} to NaN."
                )
                value = np.nan
            else:
                chrom_len = file_chroms[chrom]
                if start >= chrom_len:
                    logging.debug(
                        f"Range start ({start}) is beyond chromosome '{chrom}' length ({chrom_len}) in {file_path} "
                        f"for range {chrom}:{start}-{end}. Setting {col_name} to NaN."
                    )
                    value = np.nan
                else:
                    query_end = min(end, chrom_len)

                    if start >= query_end:
                        logging.debug(
                            f"Effective query range is invalid ({start}-{query_end}) for chromosome '{chrom}' length ({chrom_len}) "
                            f"in {file_path} for range {chrom}:{start}-{end}. Setting {col_name} to NaN."
                        )
                        value = np.nan
                    else:
                        if bw.isBigWig():
                            result_val_list = bw.stats(
                                chrom, start, query_end, type=stat_type, nBins=1
                            )

                            if (
                                result_val_list is not None
                                and result_val_list[0] is not None
                            ):
                                value = result_val_list[0]
                                logging.debug(
                                    f"Successfully got {stat_type}={value} for {chrom}:{start}-{end} from {file_path}"
                                )
                            else:
                                logging.debug(
                                    f"No data found for stat '{stat_type}' in range {chrom}:{start}-{query_end} "
                                    f"(original end: {end}) in {file_path}. Setting {col_name} to NaN."
                                )
                                value = np.nan
                        elif bw.isBigBed():
                            value = calculate_bb_stats(
                                bw, chrom, start, query_end, stat_type
                            )
                            if value is None:
                                logging.debug(
                                    f"No entries found for BigBed file {file_path} in range {chrom}:{start}-{end}. Setting {col_name} to NaN."
                                )
                                value = np.nan
                        else:
                            logging.error(
                                f"Unsupported file type for {file_path}. Expected BigWig or BigBed."
                            )
                            value = np.nan

        except Exception as e:
            logging.error(
                f"Unexpected error processing file {file_path} for range {chrom}:{start}:{end}: {e}. Setting {col_name} to NaN."
            )
            value = np.nan

        range_results[col_name] = value

    return range_results


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
    base_storage_directory=None
):
    """
    Queries multiple BigWig/BigBed files over specified genomic ranges using parallel processing.

    Manages file access using `BigWigFileManager` for efficient handling of local
    and remote files, including caching and retries. Distributes range processing
    across multiple worker threads.

    Args:
        ranges (list[tuple]): List of genomic ranges, each represented as a tuple:
                             (chromosome, start, end, name). 'name' is an identifier for the range.
        file_configs (list[dict]): List of configurations for the files to query. Each dict should have:
                                   'path': Path or URL to the BigWig/BigBed file.
                                   'stat': The statistic to calculate (e.g., 'mean', 'max', 'coverage').
                                   'name': The desired column name for this file/stat in the output.
        max_workers (int, optional): Maximum number of worker threads for parallel processing.
                                     Defaults to the number of CPU cores (`os.cpu_count()`).
        return_type (str, optional): Format for the returned results. Options: 'dataframe', 'array'.
                                     Defaults to 'dataframe'.
        url_timeout (int, optional): Timeout in seconds for downloading remote files. Defaults to 30.
        max_retries (int, optional): Maximum retry attempts for failed downloads. Defaults to 3.
        verify_ssl (bool, optional): Whether to verify SSL certificates for HTTPS URLs. Defaults to True.
        preload_files (bool, optional): If True, attempts to download and open all remote files
                                        before starting parallel processing. Defaults to True.
        keep_downloaded_files (bool, optional): If True, downloaded remote files are kept in a
                                                persistent cache. Defaults to False.
        clear_cache_on_startup (bool, optional): If True, clears the persistent cache directory
                                                 before starting the query. Defaults to False.
        base_storage_directory (str, optional): Specifies a base directory for cache and temporary files,
                                                overriding the system default.

    Returns:
        pandas.DataFrame or numpy.ndarray: The query results. Contains columns for 'chromosome',
                                           'start', 'end', 'name', and columns for each requested
                                           statistic (named according to `file_configs`). Format
                                           depends on `return_type`. Returns an empty DataFrame/array
                                           if no ranges or file configs are provided, or if no
                                           results are generated.
    """
    if not ranges:
        logging.warning("Input 'ranges' list is empty. Returning empty result.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])
    if not file_configs:
        logging.warning("Input 'file_configs' list is empty. Returning empty result.")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    results = []
    file_manager = BigWigFileManager(
        url_timeout=url_timeout, 
        max_retries=max_retries, 
        verify_ssl=verify_ssl,
        keep_downloaded_files=keep_downloaded_files,
        base_storage_directory=base_storage_directory
    )

    if clear_cache_on_startup:
        logging.info("Clearing persistent cache as requested...")
        file_manager.clear_persistent_cache()
        # If the intention was *only* to clear cache, the script might exit earlier in main().
        # Here, we assume clearing is a prelude to a query run.

    try:
        if max_workers is None:
            max_workers = os.cpu_count()
            logging.info(f"Using default max_workers: {max_workers}")
        else:
            logging.info(f"Using specified max_workers: {max_workers}")

        # Preload remote files if requested
        if preload_files:
            unique_paths = set(config["path"] for config in file_configs)
            logging.info(f"Preloading {len(unique_paths)} unique file paths")
            file_manager.preload_remote_files(unique_paths)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            worker_func = partial(
                process_single_range,
                file_configs=file_configs,
                file_manager=file_manager,
            )

            futures = [executor.submit(worker_func, r) for r in ranges]

            logging.info(
                f"Submitted {len(futures)} range queries to {max_workers} workers."
            )

            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                    if (i + 1) % 100 == 0 or (i + 1) == len(futures):
                        logging.info(f"Processed {i + 1}/{len(futures)} ranges...")
                except Exception as e:
                    logging.error(f"Error retrieving result for a range query: {e}")
    finally:
        file_manager.close_all()

    if not results:
        logging.warning(
            "No results were generated. Check logs for potential issues with ranges or files."
        )
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    logging.info(f"Collected {len(results)} results. Creating {return_type}.")

    stat_cols = [config["name"] for config in file_configs]
    column_order = ["chromosome", "start", "end", "name"] + stat_cols # Add "name" to column order

    df = pd.DataFrame(results)
    df = df.reindex(columns=column_order)

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

    Supports CSV and Parquet formats. If `output_format` is not provided,
    it attempts to infer the format from the file extension (`.csv`, `.txt` -> CSV;
    `.parquet`, `.pq` -> Parquet).

    Args:
        df (pandas.DataFrame): The DataFrame containing the results to save.
        output_path (str): The full path to the output file.
        output_format (str, optional): The desired output format ('csv' or 'parquet').
                                       If None, format is auto-detected from the extension.

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if df.empty:
        logging.error("Cannot save empty results")
        return False

    try:
        # Auto-detect format from file extension if not specified
        if output_format is None:
            _, ext = os.path.splitext(output_path.lower())
            if ext in (".csv", ".txt"):
                output_format = "csv"
            elif ext in (".parquet", ".pq"):
                output_format = "parquet"
            else:
                logging.warning(
                    f"Unable to detect format from extension '{ext}'. Defaulting to CSV."
                )
                output_format = "csv"

        # Save the file in the appropriate format
        if output_format.lower() == "csv":
            df.to_csv(output_path, index=False)
            logging.info(f"Results saved as CSV to {output_path}")
        elif output_format.lower() == "parquet":
            df.to_parquet(output_path, index=False)
            logging.info(f"Results saved as Parquet to {output_path}")
        else:
            logging.error(f"Unsupported output format: {output_format}")
            return False

        return True

    except Exception as e:
        logging.error(f"Error saving results to {output_path}: {e}")
        return False


def process_bigwig_query(
    bed_file,
    config_file,
    output_file=None,
    threads=None,
    log_level="INFO",
    return_type="dataframe",
    output_format=None,
    url_timeout=30,
    max_retries=3,
    verify_ssl=True,
    preload_files=True,
    keep_downloaded_files=False,
    clear_cache_on_startup=False,
    base_storage_directory=None
):
    """
    High-level function to coordinate the BigWig/BigBed query process.

    This function serves as the main entry point for using the script's functionality
    as a library. It handles loading configuration, reading ranges, performing
    the parallel query, and optionally saving the results.

    Args:
        bed_file (str): Path to the BED file containing genomic ranges (requires columns: chrom, start, end, name).
        config_file (str): Path to the YAML configuration file specifying files and statistics.
        output_file (str, optional): Path to save the results file. If None, results are only returned.
        threads (int, optional): Number of parallel worker threads. Defaults to `os.cpu_count()`.
        log_level (str, optional): Logging level (e.g., 'INFO', 'DEBUG'). Defaults to 'INFO'.
        return_type (str, optional): Format to return results ('dataframe' or 'array'). Defaults to 'dataframe'.
        output_format (str, optional): Format for saving results ('csv', 'parquet'). Auto-detected if None.
        url_timeout (int, optional): Timeout for URL downloads. Defaults to 30.
        max_retries (int, optional): Retries for URL downloads. Defaults to 3.
        verify_ssl (bool, optional): Verify SSL certificates for downloads. Defaults to True.
        preload_files (bool, optional): Preload remote files before processing. Defaults to True.
        keep_downloaded_files (bool, optional): Keep downloaded files in a persistent cache. Defaults to False.
        clear_cache_on_startup (bool, optional): Clear persistent cache before processing. Defaults to False.
        base_storage_directory (str, optional): Custom base directory for cache/temp files. Defaults to system temp.

    Returns:
        pandas.DataFrame or numpy.ndarray: Query results in the specified `return_type`.

    Raises:
        FileNotFoundError: If `bed_file` or `config_file` is not found.
        ValueError: If configuration is invalid or log level is incorrect.
        IOError: If there are issues reading input files or writing output files.
    """
    # Set up logging
    setup_logging(log_level)

    # Load configuration
    logging.info(f"Loading configuration from {config_file}")
    file_configs = load_config_from_yaml(config_file)

    # Read ranges from BED file
    logging.info(f"Reading genomic ranges from {bed_file}")
    ranges = read_ranges_from_bed(bed_file)

    if not ranges:
        logging.error(f"No valid ranges found in BED file: {bed_file}")
        return pd.DataFrame() if return_type == "dataframe" else np.array([])

    # Process the queries
    logging.info(f"Processing queries with {threads or os.cpu_count()} workers")
    results = query_bigwig_files(
        ranges,
        file_configs,
        max_workers=threads,
        return_type=return_type,
        url_timeout=url_timeout,
        max_retries=max_retries,
        verify_ssl=verify_ssl,
        preload_files=preload_files,
        keep_downloaded_files=keep_downloaded_files,
        clear_cache_on_startup=clear_cache_on_startup,
        base_storage_directory=base_storage_directory
    )

    # Save results if an output path was provided
    if output_file and isinstance(results, pd.DataFrame):
        save_results(results, output_file, output_format)

    return results


def main():
    """Main function to handle command-line arguments and execute the BigWig query process."""
    parser = argparse.ArgumentParser(
        description="Extract statistics from BigWig/BigBed files for genomic regions specified in a BED file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bed", "-b", required=True, help="Path to BED file containing genomic ranges"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file specifying BigWig/BigBed files and statistics",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output file (extension determines format: .csv, .parquet, .pq)",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=None, # Changed default to None, handled in query_bigwig_files
        help="Number of parallel worker threads. Defaults to the number of CPU cores.",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["csv", "parquet"],
        help="Output format (overrides file extension)",
    )
    parser.add_argument(
        "--url-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for URL connections",
    )
    parser.add_argument(
        "--url-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for URL connections",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for URL connections",
    )
    parser.add_argument(
        "--disable-preload",
        action="store_true",
        help="Disable preloading of remote files before starting workers",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded remote files in a persistent cache ({system_temp}/bwq_persistent_cache or custom base). Useful for repeated runs.",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the persistent cache of downloaded files before starting. Use with --keep-downloads if also running a query.",
    )
    parser.add_argument(
        "--base-storage-dir",
        type=str,
        default=None,
        help="Specify a base directory for storing cache and temporary files, overriding the system default."
    )

    args = parser.parse_args()

    # Setup logging as early as possible
    setup_logging(args.log_level)

    # Note: If --clear-cache is the *only* intended action, the script currently
    # still requires other arguments (bed, config, output) to proceed to process_bigwig_query.
    # For a standalone cache clear operation, the CLI parsing might need adjustment
    # or a separate script/entry point. Current implementation clears cache then proceeds if other args are valid.

    try:
        results = process_bigwig_query(
            bed_file=args.bed,
            config_file=args.config,
            output_file=args.output,
            threads=args.threads,
            log_level=args.log_level, # Logging already set up, but function might re-apply if called as lib
            output_format=args.format,
            url_timeout=args.url_timeout,
            max_retries=args.url_retries,
            verify_ssl=not args.no_verify_ssl,
            preload_files=not args.disable_preload,
            keep_downloaded_files=args.keep_downloads,
            clear_cache_on_startup=args.clear_cache,
            base_storage_directory=args.base_storage_dir # Pass CLI arg
        )

        # Check if results are empty and exit with non-zero status if so
        # Checking for DataFrame or numpy array emptiness
        if isinstance(results, pd.DataFrame) and results.empty:
            logging.warning("Processing finished, but the resulting DataFrame is empty.")
            return 1 # Indicate potential issue or no results
        elif isinstance(results, np.ndarray) and results.size == 0:
            logging.warning("Processing finished, but the resulting array is empty.")
            return 1 # Indicate potential issue or no results
        elif results is None: # Should ideally not happen unless error caught earlier
            logging.error("Processing failed to return results.")
            return 1

        # If saving was requested, process_bigwig_query already handled it.
        # The main CLI function's primary job is to parse args and call the core logic.
        logging.info("Processing completed successfully.")
        return 0

    except FileNotFoundError as e:
        logging.error(f"Input file not found: {e}")
        return 1
    except ValueError as e:
        logging.error(f"Configuration or parameter error: {e}")
        return 1
    except IOError as e:
        logging.error(f"File reading/writing error: {e}")
        return 1
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback for unexpected errors
        return 1


if __name__ == "__main__":
    sys.exit(main())
