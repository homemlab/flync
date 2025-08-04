#!/usr/bin/env python3
"""
P# Extract all feature types from appropriate inputs
p# Add MFE features to existing k-mer analysis
python orchestrator.py --# Run with configuration fpython orchestratpython orchestpython orchestrator.py --output-dir bwq_# Run feature extraction with comprehensive logging
python orchestrator.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --log-level DEBUG \
    --keep-intermediates \
    2>&1 | tee "$LOG_FILE"s/ \
    --bwq-input regulatory_regions.bed \
    --bwq-config custom_tracks.yaml \
    --bwq-threads 12 \
    --no-unify.py --output-dir mfe_screening/ \
    --mfe-input large_dataset.parquet \
    --mfe-workers 16 \
    --mfe-batch-size 10000 \
    --no-unify --output-dir kmer_analysis/ \
    --kmer-input sequences.fasta \
    --kmer-k-min 4 --kmer-k-max 16 \
    --kmer-format dense \
    --kmer-workers 24 \
    --kmer-batch-size 50000 \
    --no-unifyhon orchestrator.py \
    --config comprehensive_analysis_config.yaml \
    --output-dir config_driven_results/ut-dir incremental_results/ \
    --kmer-input /previous/analysis/kmer/kmer_features_binary_sparse.npz \
    --mfe-input new_transcript_dataset.parquet \
    --mfe-workers 8 \
    --mfe-batch-size 2000orchestrator.py --output-dir comprehensive_results/ \
    --bwq-input transcript_ranges.bed \
    --bwq-config ../../config/dm6_tracks.yaml \
    --mfe-input transcript_sequences.fasta \
    --cpat-input transcript_sequences.fasta \
    --cpat-hexamer /path/to/hexamer_table.tsv \
    --cpat-model /path/to/logit_model.pkl \
    --kmer-input transcript_sequences.fasta \
    --kmer-k-min 3 --kmer-k-max 12 \
    --log-level INFOample: Enhanced Feature Orchestrator v2
Real-world usage scenarios for comprehensive feature extraction
"""

import os
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def example_1_comprehensive_extraction():
    """
    Example 1: Comprehensive feature extraction from mixed inputs
    
    Scenario: You have different input files for different features:
    - BED file with genomic ranges for BWQ statistics
    - FASTA file with transcript sequences for MFE and k-mer
    - Existing CPAT model and hexamer table
    """
    
    logger.info("Example 1: Comprehensive Feature Extraction")
    logger.info("=" * 50)
    
    command = """
# Extract all feature types from appropriate inputs
python orchestrator.py --output-dir comprehensive_results/ \\
    --bwq-input transcript_ranges.bed \\
    --bwq-config ../../config/dm6_tracks.yaml \\
    --mfe-input transcript_sequences.fasta \\
    --cpat-input transcript_sequences.fasta \\
    --cpat-hexamer /path/to/hexamer_table.tsv \\
    --cpat-model /path/to/logit_model.pkl \\
    --kmer-input transcript_sequences.fasta \\
    --kmer-k-min 3 --kmer-k-max 12 \\
    --log-level INFO
    """
    
    logger.info("Command:")
    logger.info(command)
    
    logger.info("\nOutput structure:")
    logger.info("""
comprehensive_results/
├── bwq/
│   └── bwq_features.parquet         # BigWig/BigBed statistics
├── mfe/
│   └── mfe_features.parquet         # RNA secondary structure features
├── cpat/
│   └── cpat_features.parquet        # Coding potential scores
├── kmer/
│   ├── kmer_features_binary_sparse.npz  # Sparse k-mer matrix
│   ├── kmer_features_binary_rows.txt    # Transcript IDs
│   └── kmer_features_binary_cols.txt    # K-mer feature names
└── unified_features.parquet         # All features combined
    """)

def example_2_incremental_analysis():
    """
    Example 2: Incremental analysis with existing data
    
    Scenario: You already have k-mer features computed and want to add MFE features
    """
    
    logger.info("\nExample 2: Incremental Analysis")
    logger.info("=" * 40)
    
    command = """
# Add MFE features to existing k-mer analysis
python orchestrator.py --output-dir incremental_results/ \\
    --kmer-input /previous/analysis/kmer/kmer_features_binary_sparse.npz \\
    --mfe-input new_transcript_dataset.parquet \\
    --mfe-workers 8 \\
    --mfe-batch-size 2000
    """
    
    logger.info("Command:")
    logger.info(command)
    
    logger.info("\nUse case: Adding new feature types to existing analysis without recomputation")

def example_3_config_driven_workflow():
    """
    Example 3: Configuration-driven workflow
    
    Scenario: Complex workflow with many parameters managed via config file
    """
    
    logger.info("\nExample 3: Configuration-Driven Workflow")
    logger.info("=" * 45)
    
    # Sample configuration
    config_content = """
# comprehensive_analysis_config.yaml

# File paths
bwq_config: /project/config/dm6_tracks.yaml
hexamer_table: /project/models/hexamer_freq_table.tsv
logit_model: /project/models/cpat_logit_model.pkl
reference_fasta: /project/genomes/dm6.fa

# Feature extraction parameters
features:
  bwq:
    threads: 8
  mfe:
    workers: 12
    batch_size: 5000
  cpat:
    workers: 8
  kmer:
    k_min: 3
    k_max: 15
    output_format: sparse
    workers: 16
    batch_size: 20000

# Processing options
unify: true
use_sparse: true
keep_intermediates: false

# Input files (can be overridden via CLI)
inputs:
  bwq: /project/data/lncrna_ranges.bed
  mfe: /project/data/lncrna_sequences.fasta
  cpat: /project/data/lncrna_sequences.fasta
  kmer: /project/data/lncrna_sequences.fasta
    """
    
    logger.info("Configuration file (comprehensive_analysis_config.yaml):")
    logger.info(config_content)
    
    command = """
# Run with configuration file
python orchestrator.py \\
    --config comprehensive_analysis_config.yaml \\
    --output-dir config_driven_results/
    """
    
    logger.info("Command:")
    logger.info(command)

def example_4_targeted_feature_extraction():
    """
    Example 4: Targeted feature extraction for specific analysis
    
    Scenario: Only need specific features with custom parameters
    """
    
    logger.info("\nExample 4: Targeted Feature Extraction")
    logger.info("=" * 42)
    
    scenarios = [
        {
            "name": "High-resolution k-mer analysis",
            "command": """
python orchestrator.py --output-dir kmer_analysis/ \\
    --kmer-input sequences.fasta \\
    --kmer-k-min 4 --kmer-k-max 16 \\
    --kmer-format dense \\
    --kmer-workers 24 \\
    --kmer-batch-size 50000 \\
    --no-unify
            """
        },
        {
            "name": "Fast MFE screening",
            "command": """
python orchestrator.py --output-dir mfe_screening/ \\
    --mfe-input large_dataset.parquet \\
    --mfe-workers 16 \\
    --mfe-batch-size 10000 \\
    --no-unify
            """
        },
        {
            "name": "BWQ feature extraction with custom tracks",
            "command": """
python orchestrator.py --output-dir bwq_analysis/ \\
    --bwq-input regulatory_regions.bed \\
    --bwq-config custom_tracks.yaml \\
    --bwq-threads 12 \\
    --no-unify
            """
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"\n{scenario['name']}:")
        logger.info(scenario['command'])

def example_5_production_pipeline():
    """
    Example 5: Production pipeline with error handling and logging
    
    Scenario: Robust production workflow with comprehensive logging
    """
    
    logger.info("\nExample 5: Production Pipeline")
    logger.info("=" * 35)
    
    script_content = """#!/bin/bash
# production_feature_extraction.sh

set -euo pipefail  # Exit on any error

# Configuration
CONFIG_FILE="production_config.yaml"
OUTPUT_DIR="production_results_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${OUTPUT_DIR}/extraction.log"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run feature extraction with comprehensive logging
python orchestrator.py \\
    --config "$CONFIG_FILE" \\
    --output-dir "$OUTPUT_DIR" \\
    --log-level DEBUG \\
    --keep-intermediates \\
    2>&1 | tee "$LOG_FILE"

# Validate outputs
echo "Validating outputs..."
python validate_feature_outputs.py "$OUTPUT_DIR"

# Generate summary report
python generate_extraction_report.py \\
    --input-dir "$OUTPUT_DIR" \\
    --output "${OUTPUT_DIR}/summary_report.html"

echo "Feature extraction pipeline completed successfully!"
echo "Results in: $OUTPUT_DIR"
    """
    
    logger.info("Production script (production_feature_extraction.sh):")
    logger.info(script_content)

def demonstrate_cli_help():
    """Show the CLI help and key options"""
    
    logger.info("\nCLI Help Reference")
    logger.info("=" * 25)
    
    logger.info("""
Key commands:

# Show all options
python orchestrator.py --help

# Create sample configuration
python orchestrator.py --create-config

# Run with specific features
python orchestrator.py --output-dir results/ --mfe-input file.fasta

# Run with configuration
python orchestrator.py --config config.yaml --output-dir results/
    """)

if __name__ == "__main__":
    logger.info("Enhanced Feature Orchestrator - Practical Examples")
    logger.info("=" * 60)
    
    example_1_comprehensive_extraction()
    example_2_incremental_analysis() 
    example_3_config_driven_workflow()
    example_4_targeted_feature_extraction()
    example_5_production_pipeline()
    demonstrate_cli_help()
    
    logger.info("\n" + "=" * 60)
    logger.info("For more information, see README.md")
    logger.info("To test the functionality, run: python test_orchestrator.py")
