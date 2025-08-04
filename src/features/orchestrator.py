#!/usr/bin/env python3
"""
Enhanced Feature Extraction Orchestrator

This enhanced orchestrator supports:
- Specific inputs for each feature type (--bwq-input, --mfe-input, etc.)
- Configuration file support (--config)
- Organized output directory structure

Feature extraction modules:
- bwq.py: BigWig/BigBed statistics
- mfe_linear.py: RNA secondary structure MFE
- cpat.py: Coding potential assessment
- kmer.py: Binary k-mer profiles
"""

import os
import sys
import argparse
import logging
import pandas as pd
import tempfile
import shutil
import yaml
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import subprocess

# Import data preparation utilities with error handling
try:
    import pyranges as pr
    PYRANGES_AVAILABLE = True
except ImportError:
    PYRANGES_AVAILABLE = False

try:
    from scipy.sparse import load_npz, hstack, vstack
    import numpy as np
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the orchestrator."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        stream=sys.stdout
    )
    return logging.getLogger(__name__)


class FeatureOrchestrator:
    """Enhanced orchestrator with flexible input support and config files."""
    
    def __init__(self, output_dir: str, config_file: Optional[str] = None,
                 temp_dir: Optional[str] = None, log_level: str = 'INFO',
                 keep_intermediates: bool = False):
        """
        Initialize enhanced orchestrator.
        
        Args:
            output_dir: Output directory for all files
            config_file: Optional configuration file path
            temp_dir: Temporary directory for intermediate files
            log_level: Logging level
            keep_intermediates: Whether to keep intermediate files
        """
        # Create output directory
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logging(log_level)
        
        # Initialize attributes
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.keep_intermediates = keep_intermediates
        
        # Store paths for intermediate files
        self.intermediate_files = {}
        
        # Feature extraction script paths (relative to src/features/)
        self.script_dir = Path(__file__).parent
        self.scripts = {
            'bwq': self.script_dir / 'bwq.py',
            'mfe': self.script_dir / 'mfe_linear.py', 
            'cpat': self.script_dir / 'cpat.py',
            'kmer': self.script_dir / 'kmer.py'
        }
        
        # Validate script existence
        for name, path in self.scripts.items():
            if not path.exists():
                raise FileNotFoundError(f"Required script not found: {path}")
        
        # Load configuration if provided
        self.config = {}
        if config_file:
            self.config = self._load_config(config_file)
            self.logger.info(f"Loaded configuration from: {config_file}")
        
        # Feature-specific inputs
        self.feature_inputs = {
            'bwq': None,
            'mfe': None,
            'cpat': None,
            'kmer': None
        }
        
        # Feature-specific parameters
        self.feature_params = {
            'bwq': {},
            'mfe': {},
            'cpat': {},
            'kmer': {}
        }
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")
        except Exception as e:
            raise ValueError(f"Error loading config file {config_file}: {e}")
    
    def set_feature_input(self, feature: str, input_path: str, **params) -> None:
        """Set input and parameters for a specific feature."""
        if feature not in self.feature_inputs:
            raise ValueError(f"Unknown feature: {feature}. Valid: {list(self.feature_inputs.keys())}")
        
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        self.feature_inputs[feature] = input_path
        self.feature_params[feature].update(params)
        self.logger.info(f"Set {feature} input: {input_path}")
    
    def extract_features(self) -> Dict[str, str]:
        """
        Extract features using feature-specific inputs.
        
        Returns:
            Dictionary mapping feature names to output file paths
        """
        start_time = time.time()
        output_files = {}
        
        self.logger.info("Starting feature extraction")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Create feature-specific subdirectories
        feature_dirs = {}
        for feature in self.feature_inputs.keys():
            feature_dir = self.output_dir / feature
            feature_dir.mkdir(exist_ok=True)
            feature_dirs[feature] = feature_dir
        
        # Extract BWQ features
        if self.feature_inputs['bwq']:
            self.logger.info("Extracting BWQ features...")
            bwq_output = self._run_bwq_features(
                self.feature_inputs['bwq'],
                str(feature_dirs['bwq'] / "bwq_features.parquet"),
                **self.feature_params['bwq']
            )
            if bwq_output:
                output_files['bwq'] = bwq_output
        
        # Extract MFE features
        if self.feature_inputs['mfe']:
            self.logger.info("Extracting MFE features...")
            mfe_output = self._run_mfe_features(
                self.feature_inputs['mfe'],
                str(feature_dirs['mfe'] / "mfe_features.parquet"),
                **self.feature_params['mfe']
            )
            if mfe_output:
                output_files['mfe'] = mfe_output
        
        # Extract CPAT features
        if self.feature_inputs['cpat']:
            self.logger.info("Extracting CPAT features...")
            cpat_output = self._run_cpat_features(
                self.feature_inputs['cpat'],
                str(feature_dirs['cpat'] / "cpat_features.parquet"),
                **self.feature_params['cpat']
            )
            if cpat_output:
                output_files['cpat'] = cpat_output
        
        # Extract k-mer features
        if self.feature_inputs['kmer']:
            self.logger.info("Extracting k-mer features...")
            kmer_output = self._run_kmer_features(
                self.feature_inputs['kmer'],
                str(feature_dirs['kmer']),
                **self.feature_params['kmer']
            )
            if kmer_output:
                output_files['kmer'] = kmer_output
        
        # Unify features if multiple outputs exist
        if len(output_files) > 1 and self.feature_params.get('unify', True):
            self.logger.info("Unifying feature datasets...")
            unified_output = self._unify_features(output_files)
            if unified_output:
                output_files['unified'] = unified_output
        
        # Cleanup intermediate files
        if not self.keep_intermediates:
            for temp_file in self.intermediate_files.values():
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temporary file: {temp_file}")
        
        elapsed = time.time() - start_time
        self.logger.info(f"Feature extraction completed in {elapsed:.2f} seconds")
        
        return output_files
    
    def _run_bwq_features(self, input_file: str, output_file: str, **params) -> Optional[str]:
        """Run BWQ feature extraction with advanced parameters."""
        bwq_config = params.get('config', self.config.get('bwq_config'))
        if not bwq_config:
            self.logger.error("BWQ configuration file required but not provided")
            return None
        
        cmd = [
            'uv', 'run', 'python', str(self.scripts['bwq']),
            input_file,
            output_file,
            '--config', bwq_config,
            '--log_level', 'INFO'
        ]
        
        # Add optional parameters
        if 'threads' in params:
            cmd.extend(['--threads', str(params['threads'])])
        
        self.logger.debug(f"BWQ command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("BWQ feature extraction completed successfully")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"BWQ feature extraction failed: {e}")
            self.logger.error(f"stderr: {e.stderr}")
            return None
    
    def _run_mfe_features(self, input_file: str, output_file: str, **params) -> Optional[str]:
        """Run MFE feature extraction with advanced parameters."""
        # Handle different input formats
        processed_input = self._prepare_mfe_input(input_file)
        if not processed_input:
            return None
        
        cmd = [
            'uv', 'run', 'python', str(self.scripts['mfe']),
            processed_input,
            output_file,
            '--log_level', 'INFO'
        ]
        
        # Add optional parameters
        if 'workers' in params:
            cmd.extend(['--workers', str(params['workers'])])
        if 'batch_size' in params:
            cmd.extend(['--batch_size', str(params['batch_size'])])
        
        self.logger.debug(f"MFE command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("MFE feature extraction completed successfully")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"MFE feature extraction failed: {e}")
            self.logger.error(f"stderr: {e.stderr}")
            return None
    
    def _run_cpat_features(self, input_file: str, output_file: str, **params) -> Optional[str]:
        """Run CPAT feature extraction with advanced parameters."""
        hexamer_table = params.get('hexamer_table', self.config.get('hexamer_table'))
        logit_model = params.get('logit_model', self.config.get('logit_model'))
        
        if not hexamer_table or not logit_model:
            self.logger.error("CPAT requires hexamer table and logit model")
            return None
        
        cmd = [
            'uv', 'run', 'python', str(self.scripts['cpat']),
            input_file,
            output_file,
            '--hexamer', hexamer_table,
            '--model', logit_model,
            '--log_level', 'INFO'
        ]
        
        # Add optional parameters
        if 'workers' in params:
            cmd.extend(['--workers', str(params['workers'])])
        
        self.logger.debug(f"CPAT command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("CPAT feature extraction completed successfully")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"CPAT feature extraction failed: {e}")
            self.logger.error(f"stderr: {e.stderr}")
            return None
    
    def _run_kmer_features(self, input_file: str, output_dir: str, **params) -> Optional[str]:
        """Run k-mer feature extraction with advanced parameters."""
        output_base = str(Path(output_dir) / "kmer_features")
        
        k_min = params.get('k_min', 3)
        k_max = params.get('k_max', 12)
        output_format = params.get('output_format', 'sparse')
        
        cmd = [
            'uv', 'run', 'python', str(self.scripts['kmer']),
            '-i', input_file,
            '-o', output_base,
            '--min_k', str(k_min),
            '--max_k', str(k_max),
            '--output_format', output_format,
            '--log_level', 'INFO'
        ]
        
        # Add optional parameters
        if 'workers' in params:
            cmd.extend(['--workers', str(params['workers'])])
        if 'batch_size' in params:
            cmd.extend(['--batch_size', str(params['batch_size'])])
        
        self.logger.debug(f"K-mer command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.logger.info("K-mer feature extraction completed successfully")
            
            # Return the main output file
            if output_format == 'sparse':
                return f"{output_base}_binary_sparse.npz"
            else:
                return f"{output_base}.parquet"
        except subprocess.CalledProcessError as e:
            self.logger.error(f"K-mer feature extraction failed: {e}")
            self.logger.error(f"stderr: {e.stderr}")
            return None
    
    def _prepare_mfe_input(self, input_file: str) -> Optional[str]:
        """Prepare MFE input file (convert FASTA to CSV if needed)."""
        input_path = Path(input_file)
        
        # If already CSV/TSV/Parquet, use as-is
        if input_path.suffix.lower() in ['.csv', '.tsv', '.parquet']:
            return input_file
        
        # If FASTA, convert to CSV
        if input_path.suffix.lower() in ['.fa', '.fasta', '.fna']:
            csv_file = str(self.output_dir / f"mfe_temp_{input_path.stem}.csv")
            
            try:
                sequences = []
                with open(input_file, 'r') as f:
                    seq_id = None
                    seq_data = []
                    for line in f:
                        line = line.strip()
                        if line.startswith('>'):
                            if seq_id is not None:
                                sequences.append({'transcript_id': seq_id, 'Sequence': ''.join(seq_data)})
                            seq_id = line[1:]
                            seq_data = []
                        else:
                            seq_data.append(line)
                    # Last sequence
                    if seq_id is not None:
                        sequences.append({'transcript_id': seq_id, 'Sequence': ''.join(seq_data)})
                
                df = pd.DataFrame(sequences)
                df.to_csv(csv_file, index=False)
                self.intermediate_files['mfe_csv'] = csv_file
                return csv_file
                
            except Exception as e:
                self.logger.error(f"Failed to convert FASTA to CSV: {e}")
                return None
        
        self.logger.error(f"Unsupported MFE input format: {input_path.suffix}")
        return None
    
    def _unify_features(self, output_files: Dict[str, str]) -> Optional[str]:
        """Unify feature datasets with enhanced logic."""
        unified_output = str(self.output_dir / "unified_features.parquet")
        
        try:
            # Load all feature datasets
            feature_dfs = {}
            
            for feature_name, file_path in output_files.items():
                if feature_name == 'unified':  # Skip if already unified
                    continue
                    
                if feature_name == 'kmer' and file_path.endswith('.npz'):
                    # Handle sparse k-mer matrices
                    feature_dfs[feature_name] = self._load_sparse_kmer_data(file_path)
                else:
                    # Regular parquet files
                    feature_dfs[feature_name] = pd.read_parquet(file_path)
            
            if not feature_dfs:
                self.logger.warning("No feature datasets to unify")
                return None
            
            # Find common identifier column
            id_columns = ['transcript_id', 'sequence_id', 'id', 'name']
            common_id_col = None
            
            for col in id_columns:
                if all(col in df.columns for df in feature_dfs.values()):
                    common_id_col = col
                    break
            
            if not common_id_col:
                self.logger.error("No common identifier column found for unification")
                return None
            
            # Merge all datasets
            unified_df = None
            for feature_name, df in feature_dfs.items():
                if unified_df is None:
                    unified_df = df.copy()
                else:
                    unified_df = unified_df.merge(
                        df, on=common_id_col, how='outer', suffixes=('', f'_{feature_name}')
                    )
            
            # Save unified dataset
            unified_df.to_parquet(unified_output, index=False)
            self.logger.info(f"Unified features saved: {unified_output} ({len(unified_df)} records)")
            
            return unified_output
            
        except Exception as e:
            self.logger.error(f"Error during feature unification: {e}")
            return None
    
    def _load_sparse_kmer_data(self, npz_file: str) -> pd.DataFrame:
        """Load sparse k-mer data and convert to DataFrame."""
        if not SCIPY_AVAILABLE:
            self.logger.error("Scipy not available for loading sparse k-mer data")
            return pd.DataFrame()
        
        data = np.load(npz_file, allow_pickle=True)
        sparse_matrix = load_npz(npz_file.replace('.npz', '_sparse.npz'))
        
        # Convert to dense for merging (could be optimized for very large matrices)
        dense_matrix = sparse_matrix.toarray()
        
        # Create DataFrame with feature names
        feature_names = data['feature_names'] if 'feature_names' in data else [f'kmer_{i}' for i in range(dense_matrix.shape[1])]
        transcript_ids = data['transcript_ids'] if 'transcript_ids' in data else [f'seq_{i}' for i in range(dense_matrix.shape[0])]
        
        df = pd.DataFrame(dense_matrix, columns=feature_names)
        df.insert(0, 'transcript_id', transcript_ids)
        
        return df


def create_sample_config() -> str:
    """Create a sample configuration file."""
    config = {
        'bwq_config': '/path/to/bwq_config.yaml',
        'hexamer_table': '/path/to/hexamer_table.tsv',
        'logit_model': '/path/to/logit_model.pkl',
        'reference_fasta': '/path/to/reference.fa',
        'features': {
            'bwq': {
                'threads': 4
            },
            'mfe': {
                'workers': 4,
                'batch_size': 1000
            },
            'cpat': {
                'workers': 4
            },
            'kmer': {
                'k_min': 3,
                'k_max': 12,
                'output_format': 'sparse',
                'workers': 4,
                'batch_size': 10000
            }
        },
        'unify': True,
        'use_sparse': True,
        'keep_intermediates': False
    }
    
    config_file = 'feature_extraction_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Feature Extraction Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using configuration file
  python orchestrator.py --config config.yaml --output-dir results/

  # Specific inputs for each feature
  python orchestrator.py --output-dir results/ \\
    --bwq-input ranges.bed --bwq-config tracks.yaml \\
    --mfe-input sequences.fasta \\
    --cpat-input sequences.fasta --cpat-hexamer hex.tsv --cpat-model model.pkl \\
    --kmer-input sequences.fasta

  # Using existing k-mer sparse matrix
  python orchestrator.py --output-dir results/ \\
    --kmer-input /path/to/sparse_dir/ --mfe-input dataset.parquet

  # Create sample configuration
  python orchestrator.py --create-config
        """
    )
    
    # Main arguments
    parser.add_argument('--config', help='Configuration file (YAML or JSON)')
    parser.add_argument('--output-dir', help='Output directory for all files')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create sample configuration file and exit')
    
    # Feature-specific inputs
    parser.add_argument('--bwq-input', help='Input file for BWQ features (BED format)')
    parser.add_argument('--mfe-input', help='Input file for MFE features (FASTA/CSV/Parquet)')
    parser.add_argument('--cpat-input', help='Input file for CPAT features (FASTA)')
    parser.add_argument('--kmer-input', help='Input file/directory for k-mer features (FASTA/directory)')
    
    # BWQ parameters
    parser.add_argument('--bwq-config', help='BWQ configuration file')
    parser.add_argument('--bwq-threads', type=int, help='Number of threads for BWQ')
    
    # MFE parameters
    parser.add_argument('--mfe-workers', type=int, help='Number of workers for MFE')
    parser.add_argument('--mfe-batch-size', type=int, help='Batch size for MFE')
    
    # CPAT parameters
    parser.add_argument('--cpat-hexamer', help='Hexamer table file for CPAT')
    parser.add_argument('--cpat-model', help='Logistic model file for CPAT')
    parser.add_argument('--cpat-workers', type=int, help='Number of workers for CPAT')
    parser.add_argument('--cpat-ref', help='Reference FASTA for CPAT (alias for --cpat-hexamer)')
    
    # K-mer parameters
    parser.add_argument('--kmer-k-min', type=int, default=3, help='Minimum k-mer length')
    parser.add_argument('--kmer-k-max', type=int, default=12, help='Maximum k-mer length')
    parser.add_argument('--kmer-format', choices=['sparse', 'dense'], default='sparse',
                       help='K-mer output format')
    parser.add_argument('--kmer-workers', type=int, help='Number of workers for k-mer')
    parser.add_argument('--kmer-batch-size', type=int, help='Batch size for k-mer')
    
    # General options
    parser.add_argument('--no-unify', action='store_true', help='Skip feature unification')
    parser.add_argument('--temp-dir', help='Directory for temporary files')
    parser.add_argument('--keep-intermediates', action='store_true', 
                       help='Keep intermediate files')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Create sample config and exit
    if args.create_config:
        config_file = create_sample_config()
        print(f"Sample configuration created: {config_file}")
        print("Edit this file with your specific paths and parameters.")
        return
    
    # Check if any feature inputs provided
    if not args.output_dir:
        parser.error("--output-dir is required")
    
    feature_inputs = [args.bwq_input, args.mfe_input, args.cpat_input, args.kmer_input]
    if not any(feature_inputs) and not args.config:
        parser.error("At least one feature input or config file must be provided")
    
    try:
        # Initialize orchestrator
        orchestrator = FeatureOrchestrator(
            output_dir=args.output_dir,
            config_file=args.config,
            temp_dir=args.temp_dir,
            log_level=args.log_level,
            keep_intermediates=args.keep_intermediates
        )
        
        # Set feature inputs and parameters
        if args.bwq_input:
            bwq_params = {}
            if args.bwq_config:
                bwq_params['config'] = args.bwq_config
            if args.bwq_threads:
                bwq_params['threads'] = args.bwq_threads
            orchestrator.set_feature_input('bwq', args.bwq_input, **bwq_params)
        
        if args.mfe_input:
            mfe_params = {}
            if args.mfe_workers:
                mfe_params['workers'] = args.mfe_workers
            if args.mfe_batch_size:
                mfe_params['batch_size'] = args.mfe_batch_size
            orchestrator.set_feature_input('mfe', args.mfe_input, **mfe_params)
        
        if args.cpat_input:
            cpat_params = {}
            if args.cpat_hexamer:
                cpat_params['hexamer_table'] = args.cpat_hexamer
            if args.cpat_model:
                cpat_params['logit_model'] = args.cpat_model
            if args.cpat_ref:  # Alias support
                cpat_params['hexamer_table'] = args.cpat_ref
            if args.cpat_workers:
                cpat_params['workers'] = args.cpat_workers
            orchestrator.set_feature_input('cpat', args.cpat_input, **cpat_params)
        
        if args.kmer_input:
            kmer_params = {
                'k_min': args.kmer_k_min,
                'k_max': args.kmer_k_max,
                'output_format': args.kmer_format
            }
            if args.kmer_workers:
                kmer_params['workers'] = args.kmer_workers
            if args.kmer_batch_size:
                kmer_params['batch_size'] = args.kmer_batch_size
            orchestrator.set_feature_input('kmer', args.kmer_input, **kmer_params)
        
        # Set unification preference
        orchestrator.feature_params['unify'] = not args.no_unify
        
        # Run feature extraction
        output_files = orchestrator.extract_features()
        
        print("Feature extraction completed successfully!")
        print(f"Output directory: {args.output_dir}")
        print("Generated files:")
        for name, path in output_files.items():
            print(f"  {name}: {path}")
            
    except Exception as e:
        logging.error(f"Error during feature extraction: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
