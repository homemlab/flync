#!/usr/bin/env python3
"""
RNA Features Processing Pipeline

This script serves as a central entry point for accessing the various RNA feature
calculation tools in the package.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the directory paths to sys.path
script_dir = Path(__file__).parent
for directory in ["new-tests/bw-feature", "new-tests/mfe-feature", "new-tests/kmer-feature", "new-tests/cpat-feature"]:
    module_path = script_dir / directory
    if module_path.exists() and str(module_path) not in sys.path:
        sys.path.append(str(module_path))

# Setup logging
def setup_logging(log_level="INFO"):
    """Configure the root logger for the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{log_level}'. Defaulting to INFO.", file=sys.stderr)
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level, 
        format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        stream=sys.stdout
    )
    return logging.getLogger(__name__)

# Main entry point
def main():
    """Process arguments and call appropriate feature calculation functions."""
    parser = argparse.ArgumentParser(
        description="RNA Features Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add global arguments
    parser.add_argument('--log_level', choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help='Set the logging level')
    
    # Create subparsers for each feature
    subparsers = parser.add_subparsers(dest='feature', help='Feature to calculate')
    
    # Simple placeholder subparsers
    bwq_parser = subparsers.add_parser('bwq', help='Query BigWig/BigBed files')
    mfe_parser = subparsers.add_parser('mfe', help='Calculate minimum free energy')
    kmer_parser = subparsers.add_parser('kmer', help='Calculate k-mer profiles')
    cpat_parser = subparsers.add_parser('cpat', help='Calculate coding potential')
    
    args = parser.parse_args()
    logger = setup_logging(args.log_level)
    
    # At this point, you would typically import the specific modules and call their functions
    logger.info(f"Selected feature: {args.feature or 'None'}")
    
    if args.feature is None:
        parser.print_help()
        return 1
    
    # Import modules on demand based on selected feature
    if args.feature == 'bwq':
        try:
            from new_tests.bw_feature.bwq import process_bigwig_query
            logger.info("BigWig query module successfully imported")
        except ImportError as e:
            logger.error(f"Failed to import BigWig query module: {e}")
            return 1
            
    elif args.feature == 'mfe':
        try:
            from new_tests.mfe_feature.mfe import process_mfe_calculations
            logger.info("MFE calculation module successfully imported")
        except ImportError as e:
            logger.error(f"Failed to import MFE calculation module: {e}")
            return 1
            
    elif args.feature == 'kmer':
        try:
            from new_tests.kmer_feature.kmer import calculate_kmer_profiles
            logger.info("k-mer profiles module successfully imported")
        except ImportError as e:
            logger.error(f"Failed to import k-mer profiles module: {e}")
            return 1
            
    elif args.feature == 'cpat':
        try:
            from new_tests.cpat_feature.cpat import run_cpat_calculation
            logger.info("CPAT module successfully imported")
        except ImportError as e:
            logger.error(f"Failed to import CPAT module: {e}")
            return 1
            
    else:
        logger.error(f"Unknown feature: {args.feature}")
        return 1
        
    logger.info("Module import successful. This script would now call the appropriate function.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 