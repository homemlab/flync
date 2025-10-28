# Package Structure Recommendation

## Proposed Package Organization

To better organize the refactored code into a unified library, here's the recommended package structure:

```
flync/
├── __init__.py
├── features/
│   ├── __init__.py
│   ├── bigwig.py        # Renamed from bwq.py for clarity
│   ├── cpat.py         # Coding potential assessment
│   ├── structure.py    # Renamed from mfe_linear.py
│   └── sequence.py     # Renamed from kmer.py
├── optimization/
│   ├── __init__.py
│   ├── batch.py        # Renamed from batch_optimization.py
│   └── single.py       # Renamed from hyperparameter_optimizer.py
├── cli/
│   ├── __init__.py
│   └── commands.py     # Unified CLI entry points
└── utils/
    ├── __init__.py
    ├── data.py          # Common data loading utilities
    └── logging.py       # Common logging utilities
```

## Package-Level Imports

### flync/__init__.py
```python
"""
Flync: Feature Learning for lncrNA Classification

A comprehensive toolkit for analyzing non-coding RNA features and 
optimizing machine learning models for lncRNA classification.
"""

from .features import (
    query_bigwig_ranges,
    calculate_cpat_scores, 
    calculate_rna_mfe,
    calculate_kmers
)

from .optimization import (
    run_batch_optimization,
    optimize_hyperparameters
)

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    # Features
    'query_bigwig_ranges',
    'calculate_cpat_scores', 
    'calculate_rna_mfe',
    'calculate_kmers',
    
    # Optimization
    'run_batch_optimization', 
    'optimize_hyperparameters',
]
```

### flync/features/__init__.py
```python
"""Feature extraction and analysis tools."""

from .bigwig import query_bigwig_ranges, process_bigwig_query
from .cpat import calculate_cpat_scores
from .structure import calculate_rna_mfe  
from .sequence import calculate_kmers

__all__ = [
    'query_bigwig_ranges',
    'process_bigwig_query',
    'calculate_cpat_scores',
    'calculate_rna_mfe', 
    'calculate_kmers',
]
```

### flync/optimization/__init__.py
```python
"""Hyperparameter optimization tools."""

from .batch import run_batch_optimization
from .single import optimize_hyperparameters

__all__ = [
    'run_batch_optimization',
    'optimize_hyperparameters',
]
```

## Unified CLI Interface

### flync/cli/commands.py
```python
"""Unified command-line interface for all tools."""

import argparse
import sys
from typing import Optional

def create_main_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog='flync',
        description='Feature Learning for lncrNA Classification toolkit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # BigWig query command
    bigwig_parser = subparsers.add_parser('bigwig', help='Query BigWig/BigBed files')
    bigwig_parser.add_argument('--bed', required=True, help='BED file with ranges')
    bigwig_parser.add_argument('--config', required=True, help='YAML config file')
    bigwig_parser.add_argument('--output', required=True, help='Output file')
    bigwig_parser.add_argument('--threads', type=int, help='Number of threads')
    
    # CPAT command
    cpat_parser = subparsers.add_parser('cpat', help='Calculate coding potential')
    cpat_parser.add_argument('-i', '--input_bed', required=True, help='Input BED file')
    cpat_parser.add_argument('-r', '--reference_fasta', required=True, help='Reference genome')
    cpat_parser.add_argument('-x', '--hexamer_table', required=True, help='Hexamer table')
    cpat_parser.add_argument('-m', '--logit_model', required=True, help='Logistic model')
    cpat_parser.add_argument('-o', '--output_parquet', required=True, help='Output file')
    
    # MFE command  
    mfe_parser = subparsers.add_parser('mfe', help='Calculate RNA structure and MFE')
    mfe_parser.add_argument('input_file', help='Input file with sequences')
    mfe_parser.add_argument('output_file', help='Output file')
    mfe_parser.add_argument('--include_structure', action='store_true', help='Include structure')
    mfe_parser.add_argument('--num_processes', type=int, help='Number of processes')
    
    # K-mer command
    kmer_parser = subparsers.add_parser('kmer', help='Calculate k-mer frequencies')
    kmer_parser.add_argument('input_path', help='FASTA file or directory')
    kmer_parser.add_argument('--k_min', type=int, default=3, help='Minimum k-mer length')
    kmer_parser.add_argument('--k_max', type=int, default=12, help='Maximum k-mer length')
    kmer_parser.add_argument('--output_format', choices=['dataframe', 'matrix'], default='dataframe')
    
    # Optimization commands
    opt_parser = subparsers.add_parser('optimize', help='Hyperparameter optimization')
    opt_subparsers = opt_parser.add_subparsers(dest='opt_command')
    
    # Single optimization
    single_parser = opt_subparsers.add_parser('single', help='Single optimization run')
    single_parser.add_argument('--train-data', required=True, help='Training data')
    single_parser.add_argument('--test-data', required=True, help='Test data') 
    single_parser.add_argument('--holdout-data', required=True, help='Holdout data')
    single_parser.add_argument('--model-type', required=True, help='Model type')
    single_parser.add_argument('--study-name', required=True, help='Study name')
    single_parser.add_argument('--experiment-name', required=True, help='Experiment name')
    
    # Batch optimization
    batch_parser = opt_subparsers.add_parser('batch', help='Batch optimization')
    batch_parser.add_argument('--config', help='Configuration file')
    batch_parser.add_argument('--dry-run', action='store_true', help='Show commands only')
    
    return parser

def main() -> int:
    """Main CLI entry point."""
    parser = create_main_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'bigwig':
            from ..features.bigwig import main as bigwig_main
            # Convert args to match original interface
            sys.argv = ['bigwig', '--bed', args.bed, '--config', args.config, '--output', args.output]
            if args.threads:
                sys.argv.extend(['--threads', str(args.threads)])
            return bigwig_main()
            
        elif args.command == 'cpat':
            from ..features.cpat import main as cpat_main
            sys.argv = ['cpat', '-i', args.input_bed, '-r', args.reference_fasta,
                       '-x', args.hexamer_table, '-m', args.logit_model, 
                       '-o', args.output_parquet]
            return cpat_main()
            
        elif args.command == 'mfe':
            from ..features.structure import main as mfe_main
            sys.argv = ['mfe', args.input_file, args.output_file]
            if args.include_structure:
                sys.argv.append('--include_structure')
            if args.num_processes:
                sys.argv.extend(['--num_processes', str(args.num_processes)])
            return mfe_main()
            
        elif args.command == 'kmer':
            from ..features.sequence import main as kmer_main
            sys.argv = ['kmer', args.input_path, '--k_min', str(args.k_min), 
                       '--k_max', str(args.k_max), '--output_format', args.output_format]
            return kmer_main()
            
        elif args.command == 'optimize':
            if args.opt_command == 'single':
                from ..optimization.single import main as opt_main
                sys.argv = ['optimize', '--train-data', args.train_data,
                           '--test-data', args.test_data, '--holdout-data', args.holdout_data,
                           '--model-type', args.model_type, '--study-name', args.study_name,
                           '--experiment-name', args.experiment_name]
                return opt_main()
                
            elif args.opt_command == 'batch':
                from ..optimization.batch import main as batch_main
                sys.argv = ['batch']
                if args.config:
                    sys.argv.extend(['--config', args.config])
                if args.dry_run:
                    sys.argv.append('--dry-run')
                return batch_main()
                
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
```

## Installation and Usage

### As a Package

After reorganizing into the package structure:

```bash
# Install in development mode
pip install -e .

# Use the unified CLI
flync bigwig --bed ranges.bed --config tracks.yaml --output results.csv
flync cpat -i transcripts.bed -r genome.fa -x hexamer.tsv -m model.RData -o results.parquet
flync mfe sequences.csv results.csv --include_structure
flync kmer sequences.fasta --k_min 3 --k_max 8
flync optimize single --train-data train.parquet --test-data test.parquet --holdout-data holdout.parquet --model-type randomforest --study-name study1 --experiment-name exp1
flync optimize batch --config batch_config.yaml
```

### As a Library

```python
import flync

# Feature extraction
bigwig_results = flync.query_bigwig_ranges("ranges.bed", "tracks.yaml")
cpat_results = flync.calculate_cpat_scores("transcripts.bed", "genome.fa", "hexamer.tsv", "model.RData")
mfe_results = flync.calculate_rna_mfe("sequences.csv")
kmer_results = flync.calculate_kmers("sequences.fasta")

# Optimization
batch_results = flync.run_batch_optimization(config_path="batch_config.yaml")
single_results = flync.optimize_hyperparameters(
    train_data_path="train.parquet",
    test_data_path="test.parquet", 
    holdout_data_path="holdout.parquet",
    model_type="randomforest",
    study_name="study1",
    experiment_name="exp1"
)
```

## Setup.py Configuration

```python
from setuptools import setup, find_packages

setup(
    name="flync",
    version="1.0.0",
    description="Feature Learning for lncrNA Classification",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/flync",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'flync=flync.cli.commands:main',
        ],
    },
    install_requires=[
        "pandas",
        "numpy", 
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "pyranges",
        "biopython",
        "optuna",
        "mlflow",
        "lightgbm",
        "xgboost",
    ],
    extras_require={
        "bigwig": ["pyBigWig"],
        "cpat": ["cpmodule"], 
        "structure": [],  # LinearFold external dependency
        "optimization": ["interpret"],
        "dev": ["pytest", "black", "flake8", "mypy"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
```

This package structure provides:

1. **Clean separation of concerns** - features vs optimization
2. **Unified CLI interface** - single entry point for all tools
3. **Easy imports** - `import flync` gives access to all functionality
4. **Backward compatibility** - original CLI scripts still work
5. **Extensibility** - easy to add new features or optimization methods
6. **Professional packaging** - ready for PyPI distribution
