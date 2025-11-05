"""
Public Python API for FLYNC Pipeline

This module provides high-level Python functions for running the FLYNC pipeline
programmatically, allowing integration into larger workflows or scripts.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def run_pipeline(
    config_path: Path,
    cores: int = 8,
    ml_threads: int = 8,
    dry_run: bool = False,
    skip_bio: bool = False,
    skip_ml: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete FLYNC pipeline from Python code.
    
    This function mirrors the functionality of the `flync run-all` CLI command,
    orchestrating both the bioinformatics and ML prediction stages.
    
    Parameters
    ----------
    config_path : Path
        Path to unified pipeline configuration YAML file
    cores : int, optional
        Number of cores for bioinformatics pipeline (default: 8)
    ml_threads : int, optional
        Number of threads for ML feature extraction (default: 8)
    dry_run : bool, optional
        If True, show what would be done without executing (default: False)
    skip_bio : bool, optional
        If True, skip bioinformatics pipeline (default: False)
    skip_ml : bool, optional
        If True, skip ML prediction pipeline (default: False)
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing pipeline results and status:
        - 'status': 'success' or 'failed'
        - 'bio_completed': bool
        - 'ml_completed': bool
        - 'output_dir': Path
        - 'predictions_file': Path (if ML completed)
        - 'error': str (if failed)
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    ValueError
        If config file is missing required keys
    RuntimeError
        If pipeline execution fails
    
    Examples
    --------
    Run complete pipeline:
    
    >>> from flync.api import run_pipeline
    >>> from pathlib import Path
    >>> 
    >>> result = run_pipeline(
    ...     config_path=Path("config.yaml"),
    ...     cores=16,
    ...     ml_threads=8
    ... )
    >>> print(f"Status: {result['status']}")
    >>> print(f"Predictions: {result['predictions_file']}")
    
    Run only bioinformatics:
    
    >>> result = run_pipeline(
    ...     config_path=Path("config.yaml"),
    ...     skip_ml=True
    ... )
    
    Run only ML prediction (using existing GTF):
    
    >>> result = run_pipeline(
    ...     config_path=Path("config.yaml"),
    ...     skip_bio=True
    ... )
    """
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}")
    
    # Validate required config keys
    required_keys = ['output_dir']
    if not skip_bio:
        required_keys.extend(['genome', 'annotation', 'hisat_index'])
    if not skip_ml:
        required_keys.extend(['ml_reference_genome', 'ml_output_file'])
    
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required config keys: {', '.join(missing_keys)}\n"
            f"Required keys: {', '.join(required_keys)}"
        )
    
    output_dir = Path(config['output_dir'])
    result = {
        'status': 'running',
        'bio_completed': False,
        'ml_completed': False,
        'output_dir': output_dir,
    }
    
    if verbose:
        print("=" * 60)
        print("FLYNC Pipeline (Python API)")
        print("=" * 60)
        print(f"Configuration: {config_path}")
        print(f"Bioinformatics cores: {cores}")
        print(f"ML threads: {ml_threads}")
        print()
    
    # Phase 1: Bioinformatics Pipeline
    if not skip_bio:
        if verbose:
            print("\n[Phase 1/2] Running bioinformatics pipeline...")
            print("-" * 60)
        
        try:
            # Find the Snakefile within the package
            snakefile_path = pkg_resources.files("flync.workflows").joinpath("Snakefile")
            
            cmd = [
                "snakemake",
                "--snakefile",
                str(snakefile_path),
                "--configfile",
                str(config_path),
                "--cores",
                str(cores),
                "--use-conda",
                "--rerun-incomplete",
            ]
            
            if dry_run:
                cmd.extend(["--dry-run", "--printshellcmds"])
            
            if verbose:
                print(f"Executing: {' '.join(cmd)}\n")
            
            result_bio = subprocess.run(
                cmd,
                check=not dry_run,
                capture_output=not verbose
            )
            
            if not dry_run and result_bio.returncode != 0:
                raise RuntimeError(
                    f"Bioinformatics pipeline failed with error code {result_bio.returncode}"
                )
            
            result['bio_completed'] = True
            
            if verbose:
                print("\n✓ Bioinformatics pipeline completed!")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = f"Bioinformatics pipeline error: {str(e)}"
            raise RuntimeError(result['error'])
    else:
        if verbose:
            print("\n[Phase 1/2] Skipping bioinformatics pipeline")
    
    # Phase 2: ML Prediction Pipeline
    if not skip_ml:
        if verbose:
            print("\n[Phase 2/2] Running ML prediction pipeline...")
            print("-" * 60)
        
        if dry_run:
            if verbose:
                print("(Dry run mode - would run ML prediction)")
                print(f"  Input GTF: {output_dir / 'assemblies/merged-new-transcripts.gtf'}")
                print(f"  Output: {config.get('ml_output_file', 'predictions.csv')}")
            result['status'] = 'success'
            return result
        
        try:
            from flync.ml.predictor import predict_lncrna
            
            # Determine GTF file to use
            gtf_file = config.get('ml_gtf')
            if gtf_file is None:
                gtf_file = str(output_dir / 'assemblies' / 'merged-new-transcripts.gtf')
                if verbose:
                    print(f"Auto-detected GTF: {gtf_file}")
            
            if not Path(gtf_file).exists():
                raise FileNotFoundError(
                    f"GTF file not found: {gtf_file}. "
                    "Ensure bioinformatics pipeline completed successfully or provide ml_gtf in config"
                )
            
            # Get ML parameters from config
            ref_genome = config['ml_reference_genome']
            output_file = config['ml_output_file']
            bwq_config = config.get('ml_bwq_config')
            model_file = config.get('ml_model')
            cache_dir = config.get('ml_cache_dir')
            
            if verbose:
                print(f"  Input GTF: {gtf_file}")
                print(f"  Reference genome: {ref_genome}")
                print(f"  Output: {output_file}")
                if bwq_config:
                    print(f"  BWQ config: {bwq_config}")
                if model_file:
                    print(f"  Model: {model_file}")
                print()
            
            # Run prediction
            predict_lncrna(
                gtf_file=gtf_file,
                model_file=model_file,
                output_file=output_file,
                ref_genome=ref_genome,
                bwq_config=bwq_config,
                threads=ml_threads,
                cache_dir=cache_dir,
                clear_cache=False,
                verbose=verbose,
            )
            
            result['ml_completed'] = True
            result['predictions_file'] = Path(output_file)
            
            if verbose:
                print("\n✓ ML prediction completed!")
            
        except Exception as e:
            result['status'] = 'failed'
            result['error'] = f"ML prediction error: {str(e)}"
            raise RuntimeError(result['error'])
    else:
        if verbose:
            print("\n[Phase 2/2] Skipping ML prediction")
    
    # Final summary
    result['status'] = 'success'
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Complete FLYNC pipeline finished successfully!")
        print("=" * 60)
        print("\nResults:")
        if not skip_bio:
            print(f"  Bioinformatics: {output_dir}")
            print(f"    - Assemblies: {output_dir / 'assemblies'}")
            print(f"    - Quantification: {output_dir / 'cov'}")
            if (output_dir / 'dge').exists():
                print(f"    - DGE analysis: {output_dir / 'dge'}")
        if not skip_ml:
            print(f"  ML predictions: {config['ml_output_file']}")
        print()
    
    return result


def run_bioinformatics(
    config_path: Path,
    cores: int = 8,
    dry_run: bool = False,
    unlock: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run only the bioinformatics pipeline from Python code.
    
    This function mirrors `flync run-bio` command functionality.
    
    Parameters
    ----------
    config_path : Path
        Path to pipeline configuration YAML file
    cores : int, optional
        Number of cores/threads to use (default: 8)
    dry_run : bool, optional
        Perform dry run without execution (default: False)
    unlock : bool, optional
        Unlock working directory after crash (default: False)
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'status', 'output_dir', and optional 'error'
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    RuntimeError
        If pipeline execution fails
    """
    
    return run_pipeline(
        config_path=config_path,
        cores=cores,
        dry_run=dry_run,
        skip_ml=True,
        verbose=verbose,
    )


def run_ml_prediction(
    gtf_file: Path,
    output_file: Path,
    ref_genome: Path,
    model_file: Optional[Path] = None,
    bwq_config: Optional[Path] = None,
    threads: int = 8,
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run only ML prediction from Python code.
    
    This function mirrors `flync run-ml` command functionality.
    
    Parameters
    ----------
    gtf_file : Path
        Input GTF file with transcripts to classify
    output_file : Path
        Output CSV file for predictions
    ref_genome : Path
        Reference genome FASTA file
    model_file : Path, optional
        Path to trained model (uses bundled model if None)
    bwq_config : Path, optional
        BigWig query configuration file
    threads : int, optional
        Number of threads for feature extraction (default: 8)
    cache_dir : Path, optional
        Directory for caching genomic tracks
    verbose : bool, optional
        Print progress messages (default: True)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'status', 'predictions_file', 'n_lncrna', 'n_coding'
    
    Raises
    ------
    FileNotFoundError
        If required input files don't exist
    RuntimeError
        If prediction fails
    """
    
    from flync.ml.predictor import predict_lncrna
    import pandas as pd
    
    result = {'status': 'running'}
    
    try:
        if verbose:
            print("Running ML prediction...")
            print(f"  Input GTF: {gtf_file}")
            print(f"  Output: {output_file}")
        
        predict_lncrna(
            gtf_file=str(gtf_file),
            model_file=str(model_file) if model_file else None,
            output_file=str(output_file),
            ref_genome=str(ref_genome),
            bwq_config=str(bwq_config) if bwq_config else None,
            threads=threads,
            cache_dir=str(cache_dir) if cache_dir else None,
            verbose=verbose,
        )
        
        # Read results to get counts
        df = pd.read_csv(output_file)
        n_lncrna = int((df['prediction'] == 1).sum())
        n_coding = int((df['prediction'] == 0).sum())
        
        result['status'] = 'success'
        result['predictions_file'] = Path(output_file)
        result['n_lncrna'] = n_lncrna
        result['n_coding'] = n_coding
        
        if verbose:
            print(f"\n✓ Predicted {n_lncrna} lncRNAs, {n_coding} coding transcripts")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        raise RuntimeError(f"ML prediction error: {str(e)}")
    
    return result
