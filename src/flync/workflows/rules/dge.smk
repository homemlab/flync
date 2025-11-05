"""
Differential Gene Expression Analysis Rules

This module performs DGE analysis using Ballgown when metadata with condition
information is provided. The DGE rule only runs if:
  1. samples config points to a metadata.csv file (not a simple .txt file)
  2. The metadata CSV contains a 'condition' column for grouping
"""

from pathlib import Path

# Check if we should run DGE based on configuration
samples_path_config = config.get("samples", None)
RUN_DGE = False

if samples_path_config is not None:
    samples_path = Path(samples_path_config)
    if not samples_path.is_absolute():
        # Try relative to working directory or config file directory
        if not samples_path.exists() and workflow.configfiles:
            config_dir = Path(workflow.configfiles[0]).parent
            samples_path = config_dir / samples_path
    
    # Only run DGE if samples file is CSV (not TXT)
    if samples_path.exists() and samples_path.suffix in ['.csv', '.CSV']:
        RUN_DGE = True
        METADATA_CSV = str(samples_path.resolve())
        print(f"DGE analysis enabled: metadata file = {METADATA_CSV}")
    else:
        print(f"DGE analysis disabled: samples file is not CSV or doesn't exist")
else:
    print("DGE analysis disabled: no samples file specified")


rule ballgown_dge:
    """
    Perform differential gene expression analysis using Ballgown.
    
    This rule runs only when:
      - samples config points to a metadata.csv file
      - Metadata contains a 'condition' column for comparing groups
    
    Input:
      - StringTie quantification outputs for all samples (from quantify rule)
      - Metadata CSV with sample_id and condition columns
    
    Output:
      - transcript_dge_results.csv: Transcript-level DE results
      - gene_dge_results.csv: Gene-level DE results
      - dge_summary.csv: Summary statistics
      - ballgown_dge.log: Analysis log
    """
    input:
        # Ensure all quantification is complete before running DGE
        quant_flag = OUTPUT_DIR / "cov/quantification_complete.txt",
        metadata = METADATA_CSV if RUN_DGE else [],
    output:
        transcript_results = OUTPUT_DIR / "dge/transcript_dge_results.csv",
        gene_results = OUTPUT_DIR / "dge/gene_dge_results.csv",
        summary = OUTPUT_DIR / "dge/dge_summary.csv",
        log_file = OUTPUT_DIR / "dge/ballgown_dge.log",
    params:
        cov_dir = OUTPUT_DIR / "cov",
        output_dir = OUTPUT_DIR / "dge",
        script = lambda wildcards: Path(workflow.basedir) / "scripts" / "ballgown_dge.R",
    log:
        OUTPUT_DIR / "logs/dge/ballgown_dge.log"
    threads: 1
    shell:
        """
        set -e
        echo "Running Ballgown DGE analysis..."
        echo "  Coverage directory: {params.cov_dir}"
        echo "  Metadata file: {input.metadata}"
        echo "  Output directory: {params.output_dir}"
        
        # Run R script
        Rscript {params.script} \
            {params.cov_dir} \
            {input.metadata} \
            {params.output_dir} \
            > {log} 2>&1
        
        echo "DGE analysis complete"
        """
