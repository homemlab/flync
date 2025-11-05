"""
FLYNC Command Line Interface

Python-first CLI for the FLYNC lncRNA discovery pipeline.
"""

import click
import subprocess
import sys
import os
from pathlib import Path
from importlib import resources
import yaml

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


class FlyncGroup(click.Group):
    """Custom Click group with better error messages"""

    def parse_args(self, ctx, args):
        """Override to provide helpful error messages for common mistakes"""
        # Allow --help and --version to pass through
        if args and args[0] in ("--help", "-h", "--version", "-v"):
            return super().parse_args(ctx, args)

        # Check if user provided options without a command
        if args and args[0].startswith("-"):
            # Check if it looks like run-ml options
            if "-g" in args or "--gtf" in args:
                click.echo(
                    "Error: Missing command. Did you mean 'flync run-ml'?", err=True
                )
                click.echo(
                    "\nYou provided options for the ML pipeline, but forgot the 'run-ml' command.",
                    err=True,
                )
                click.echo("\nCorrect usage:", err=True)
                click.echo(
                    "  flync run-ml -g <gtf_file> -o <output_file> -r <genome_fasta> [options]",
                    err=True,
                )
                click.echo("\nFor more help, run: flync run-ml --help", err=True)
                ctx.exit(2)
            elif "-c" in args or "--configfile" in args or "--cores" in args:
                click.echo(
                    "Error: Missing command. Did you mean 'flync run-bio'?", err=True
                )
                click.echo(
                    "\nYou provided options for the bioinformatics pipeline, but forgot the 'run-bio' command.",
                    err=True,
                )
                click.echo("\nCorrect usage:", err=True)
                click.echo("  flync run-bio -c <config.yaml> [options]", err=True)
                click.echo("\nFor more help, run: flync run-bio --help", err=True)
                ctx.exit(2)
            else:
                click.echo("Error: Missing command.", err=True)
                click.echo("\nAvailable commands:", err=True)
                click.echo(
                    "  flync run-all  - Run complete pipeline (bioinformatics + ML)",
                    err=True,
                )
                click.echo(
                    "  flync run-bio  - Run bioinformatics assembly pipeline", err=True
                )
                click.echo(
                    "  flync run-ml   - Run ML lncRNA prediction pipeline", err=True
                )
                click.echo(
                    "  flync setup    - Download genome and build indices", err=True
                )
                click.echo(
                    "  flync config   - Generate configuration template", err=True
                )
                click.echo("\nFor more help, run: flync --help", err=True)
                ctx.exit(2)

        return super().parse_args(ctx, args)


@click.group(cls=FlyncGroup, invoke_without_command=True)
@click.option("--version", is_flag=True, help="Show version and exit")
@click.pass_context
def main(ctx, version):
    """
    FLYNC: lncRNA discovery pipeline for Drosophila melanogaster

    A bioinformatics pipeline for discovering and classifying non-coding genes.
    Combines RNA-seq processing, feature extraction from genomic databases,
    and machine learning prediction.

    \b
    Available commands:
      setup     Download genome and build indices
      config    Generate configuration template
      run-bio   Run bioinformatics assembly pipeline
      run-ml    Run ML lncRNA prediction pipeline
      run-all   Run complete pipeline (bioinformatics + ML)

    \b
    Examples:
      flync setup --genome-dir genome
      flync config --template --output config.yaml
      flync run-bio -c config.yaml -j 8
      flync run-ml -g merged.gtf -o predictions.csv -r genome.fa
      flync run-all -c config.yaml -j 8
    """
    if version:
        try:
            from importlib.metadata import version as get_version

            pkg_version = get_version("flync")
        except Exception:
            pkg_version = "1.0.0"
        click.echo(f"FLYNC version {pkg_version}")
        ctx.exit(0)

    # Show help if no command provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(0)


@main.command("run-bio")
@click.option(
    "--configfile",
    "-c",
    default="config/config.yaml",
    type=click.Path(exists=True),
    help="Path to pipeline configuration file",
)
@click.option(
    "--cores",
    "-j",
    default=8,
    type=int,
    help="Number of cores/threads to use",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Perform a dry run (don't execute, just show what would be done)",
)
@click.option(
    "--unlock",
    is_flag=True,
    help="Unlock the working directory (useful after a crash)",
)
def run_bio(configfile, cores, dry_run, unlock):
    """
    Run the bioinformatics transcriptome assembly pipeline.

    This command executes the complete RNA-seq analysis workflow:
    - Read mapping with HISAT2
    - Transcriptome assembly with StringTie
    - Assembly merging and comparison
    - Transcript quantification

    Configure input mode (SRA vs local FASTQ) in your config.yaml file:
    - For SRA: provide 'samples' CSV/TXT file
    - For local FASTQ: set 'fastq_dir' and 'fastq_paired' in config
    - For auto-detection: set 'samples: null' and 'fastq_dir: /path/to/fastq'
    """
    click.echo("Starting bioinformatics pipeline...")
    click.echo(f"  Configuration: {configfile}")
    click.echo(f"  Cores: {cores}")

    try:
        # Find the Snakefile within the package
        snakefile_path = pkg_resources.files("flync.workflows").joinpath("Snakefile")

        cmd = [
            "snakemake",
            "--snakefile",
            str(snakefile_path),
            "--configfile",
            configfile,
            "--cores",
            str(cores),
            "--use-conda",
            "--rerun-incomplete",
        ]

        if dry_run:
            cmd.append("--dry-run")
            cmd.append("--printshellcmds")

        if unlock:
            cmd.append("--unlock")

        click.echo(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True)

        click.secho("✓ Pipeline completed successfully!", fg="green", bold=True)

    except subprocess.CalledProcessError as e:
        click.secho(
            f"✗ Pipeline failed with error code {e.returncode}", fg="red", bold=True
        )
        sys.exit(e.returncode)
    except Exception as e:
        click.secho(f"✗ Error: {str(e)}", fg="red", bold=True)
        sys.exit(1)


@main.command("run-ml")
@click.option(
    "--gtf",
    "-g",
    required=True,
    type=click.Path(exists=True),
    help="Input GTF file (e.g., merged.gtf or merged-new-transcripts.gtf)",
)
@click.option(
    "--model",
    "-m",
    type=click.Path(exists=True),
    help="Path to trained ML model file (if not provided, uses bundled model)",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output file for lncRNA predictions",
)
@click.option(
    "--ref-genome",
    "-r",
    type=click.Path(exists=True),
    help="Path to reference genome FASTA file",
)
@click.option(
    "--bwq-config",
    type=click.Path(exists=True),
    help="Path to BigWig query configuration file",
)
@click.option(
    "--threads",
    "-t",
    default=8,
    type=int,
    help="Number of threads for feature extraction",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    help="Directory for caching downloaded genomic tracks (default: system temp directory)",
)
@click.option(
    "--clear-cache",
    is_flag=True,
    help="Clear the cache directory before starting",
)
@click.option(
    "--cov-dir",
    type=click.Path(exists=True),
    help="Directory containing coverage GTF files (<sample_id>.rna.gtf). If not provided, infers from input GTF path.",
)
def run_ml(
    gtf, model, output, ref_genome, bwq_config, threads, cache_dir, clear_cache, cov_dir
):
    """
    Run the lncRNA prediction ML pipeline.

    This command performs:
    - Feature extraction from GTF and genomic tracks
    - ML-based lncRNA classification
    - Output of predictions with confidence scores

    Feature Extraction:
    - K-mer features (3-12mers) with TF-IDF and SVD dimensionality reduction
    - BigWig track quantification (chromatin marks, conservation, etc.)
    - RNA secondary structure features
    - Cached genomic tracks for faster reruns
    """
    click.echo("Starting ML inference pipeline...")
    click.echo(f"  Input GTF: {gtf}")
    click.echo(f"  Output: {output}")

    try:
        # Import ML modules (will be created in next step)
        from flync.ml.predictor import predict_lncrna

        # Use bundled model if not provided
        if model is None:
            # Use __file__ to reliably locate assets directory
            import flync

            flync_dir = Path(flync.__file__).parent
            model_path = flync_dir / "assets" / "flync_ebm_model.pkl"

            if not model_path.exists():
                # Fallback: check if running from source
                src_path = Path(__file__).parent / "assets" / "flync_ebm_model.pkl"
                if src_path.exists():
                    model_path = src_path
                else:
                    raise FileNotFoundError(
                        f"Cannot find bundled model at {model_path} or {src_path}"
                    )

            model = str(model_path)
            click.echo(f"  Using bundled model: {model}")

        # Run prediction
        predict_lncrna(
            gtf_file=gtf,
            model_file=model,
            output_file=output,
            ref_genome=ref_genome,
            bwq_config=bwq_config,
            threads=threads,
            cache_dir=cache_dir,
            clear_cache=clear_cache,
            cov_dir=cov_dir,
        )

        click.secho("✓ ML prediction completed successfully!", fg="green", bold=True)
        click.echo(f"  Results saved to: {output}")

    except ImportError as e:
        click.secho(f"✗ Error importing ML modules: {str(e)}", fg="red", bold=True)
        click.echo("  Note: ML modules are being migrated. Please check back soon.")
        sys.exit(1)
    except Exception as e:
        click.secho(f"✗ Prediction failed: {str(e)}", fg="red", bold=True)
        sys.exit(1)


@main.command("run-all")
@click.option(
    "--configfile",
    "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to unified pipeline configuration file",
)
@click.option(
    "--cores",
    "-j",
    default=8,
    type=int,
    help="Number of cores/threads to use for bioinformatics pipeline",
)
@click.option(
    "--ml-threads",
    "-t",
    default=8,
    type=int,
    help="Number of threads for ML feature extraction",
)
@click.option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Perform a dry run (don't execute, just show what would be done)",
)
@click.option(
    "--skip-bio",
    is_flag=True,
    help="Skip bioinformatics pipeline (use existing GTF)",
)
@click.option(
    "--skip-ml",
    is_flag=True,
    help="Skip ML prediction (only run bioinformatics pipeline)",
)
def run_all(configfile, cores, ml_threads, dry_run, skip_bio, skip_ml):
    """
    Run the complete FLYNC pipeline end-to-end.

    This command orchestrates the entire lncRNA discovery workflow:
    1. Bioinformatics pipeline (flync run-bio):
       - Read mapping, assembly, merging, quantification
       - Optional DGE analysis if metadata CSV provided
    2. ML prediction pipeline (flync run-ml):
       - Feature extraction from assembled transcripts
       - lncRNA classification with trained EBM model

    The config file should contain both bioinformatics and ML parameters.
    Required config keys:
      - Bioinformatics: samples, genome, annotation, hisat_index, output_dir
      - ML: ml_reference_genome, ml_output_file, ml_gtf (optional, auto-detected)
      - Optional: ml_bwq_config, ml_cache_dir, ml_model

    Example config.yaml:
        samples: metadata.csv
        genome: genome/genome.fa
        annotation: genome/genome.gtf
        hisat_index: genome/genome.idx
        output_dir: results
        threads: 8
        ml_reference_genome: genome/genome.fa
        ml_output_file: predictions.csv
        ml_bwq_config: config/bwq_config.yaml  # optional
    """
    import yaml

    click.echo("=" * 60)
    click.echo("FLYNC Complete Pipeline")
    click.echo("=" * 60)
    click.echo(f"Configuration: {configfile}")
    click.echo(f"Bioinformatics cores: {cores}")
    click.echo(f"ML threads: {ml_threads}")
    click.echo()

    # Load configuration
    try:
        with open(configfile, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        click.secho(f"✗ Failed to load config file: {e}", fg="red", bold=True)
        sys.exit(1)

    # Validate required config keys
    required_keys = ["output_dir"]
    if not skip_bio:
        required_keys.extend(["genome", "annotation", "hisat_index"])
    if not skip_ml:
        required_keys.extend(["ml_reference_genome", "ml_output_file"])

    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        click.secho(
            f"✗ Missing required config keys: {', '.join(missing_keys)}",
            fg="red",
            bold=True,
        )
        click.echo("\nRequired keys for run-all:")
        click.echo("  - output_dir")
        if not skip_bio:
            click.echo("  - genome, annotation, hisat_index")
        if not skip_ml:
            click.echo("  - ml_reference_genome, ml_output_file")
        sys.exit(1)

    output_dir = Path(config["output_dir"])

    # Phase 1: Bioinformatics Pipeline
    if not skip_bio:
        click.echo("\n[Phase 1/2] Running bioinformatics pipeline...")
        click.echo("-" * 60)

        if dry_run:
            click.echo("(Dry run mode - no actual execution)")

        try:
            # Find the Snakefile within the package
            snakefile_path = pkg_resources.files("flync.workflows").joinpath(
                "Snakefile"
            )

            cmd = [
                "snakemake",
                "--snakefile",
                str(snakefile_path),
                "--configfile",
                configfile,
                "--cores",
                str(cores),
                "--use-conda",
                "--rerun-incomplete",
            ]

            if dry_run:
                cmd.append("--dry-run")
                cmd.append("--printshellcmds")

            click.echo(f"Executing: {' '.join(cmd)}\n")
            result = subprocess.run(cmd, check=not dry_run)

            if not dry_run and result.returncode != 0:
                click.secho(
                    f"✗ Bioinformatics pipeline failed with error code {result.returncode}",
                    fg="red",
                    bold=True,
                )
                sys.exit(result.returncode)

            click.secho("✓ Bioinformatics pipeline completed!", fg="green", bold=True)

        except Exception as e:
            click.secho(
                f"✗ Bioinformatics pipeline error: {str(e)}", fg="red", bold=True
            )
            sys.exit(1)
    else:
        click.echo("\n[Phase 1/2] Skipping bioinformatics pipeline (--skip-bio)")

    # Phase 2: ML Prediction Pipeline
    if not skip_ml:
        click.echo("\n[Phase 2/2] Running ML prediction pipeline...")
        click.echo("-" * 60)

        if dry_run:
            click.echo("(Dry run mode - would run ML prediction)")
            click.echo(
                f"  Input GTF: {output_dir / 'assemblies/merged-new-transcripts.gtf'}"
            )
            click.echo(f"  Output: {config.get('ml_output_file', 'predictions.csv')}")
            click.secho(
                "\n✓ Pipeline orchestration complete (dry run)", fg="green", bold=True
            )
            return

        try:
            from flync.ml.predictor import predict_lncrna

            # Determine GTF file to use
            # Priority: config['ml_gtf'] > output_dir/assemblies/merged-new-transcripts.gtf
            gtf_file = config.get("ml_gtf")
            if gtf_file is None:
                gtf_file = str(output_dir / "assemblies" / "merged-new-transcripts.gtf")
                click.echo(f"Auto-detected GTF: {gtf_file}")

            if not Path(gtf_file).exists():
                click.secho(f"✗ GTF file not found: {gtf_file}", fg="red", bold=True)
                click.echo(
                    "Ensure bioinformatics pipeline completed successfully or provide ml_gtf in config"
                )
                sys.exit(1)

            # Get ML parameters from config
            ref_genome = config["ml_reference_genome"]
            output_file = config["ml_output_file"]
            bwq_config = config.get("ml_bwq_config")
            model_file = config.get("ml_model")
            cache_dir = config.get("ml_cache_dir")

            # Use bundled model if not provided
            if model_file is None:
                import flync

                flync_dir = Path(flync.__file__).parent
                model_path = flync_dir / "assets" / "flync_ebm_model.pkl"

                if not model_path.exists():
                    # Fallback: check if running from source
                    src_path = Path(__file__).parent / "assets" / "flync_ebm_model.pkl"
                    if src_path.exists():
                        model_path = src_path
                    else:
                        raise FileNotFoundError(
                            f"Cannot find bundled model at {model_path} or {src_path}"
                        )

                model_file = str(model_path)

            click.echo(f"  Input GTF: {gtf_file}")
            click.echo(f"  Reference genome: {ref_genome}")
            click.echo(f"  Output: {output_file}")
            if bwq_config:
                click.echo(f"  BWQ config: {bwq_config}")
            click.echo(f"  Model: {model_file}")
            click.echo()

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
                verbose=True,
            )

            click.secho("✓ ML prediction completed!", fg="green", bold=True)

        except Exception as e:
            click.secho(f"✗ ML prediction error: {str(e)}", fg="red", bold=True)
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        click.echo("\n[Phase 2/2] Skipping ML prediction (--skip-ml)")

    # Final summary
    click.echo("\n" + "=" * 60)
    click.secho(
        "✓ Complete FLYNC pipeline finished successfully!", fg="green", bold=True
    )
    click.echo("=" * 60)
    click.echo("\nResults:")
    if not skip_bio:
        click.echo(f"  Bioinformatics: {output_dir}")
        click.echo(f"    - Assemblies: {output_dir / 'assemblies'}")
        click.echo(f"    - Quantification: {output_dir / 'cov'}")
        if (output_dir / "dge").exists():
            click.echo(f"    - DGE analysis: {output_dir / 'dge'}")
    if not skip_ml:
        click.echo(f"  ML predictions: {config['ml_output_file']}")
    click.echo()


@main.command("setup")
@click.option(
    "--genome-dir",
    "-d",
    default="genome",
    type=click.Path(),
    help="Directory to store genome files",
)
@click.option(
    "--skip-download",
    is_flag=True,
    help="Skip genome download if files already exist",
)
@click.option(
    "--build-index",
    is_flag=True,
    default=True,
    help="Build HISAT2 index after download",
)
def setup(genome_dir, skip_download, build_index):
    """
    Download reference genome and build indices.

    Downloads Drosophila melanogaster BDGP6.32 (dm6) genome and annotation
    from Ensembl release 106, then builds HISAT2 indices.
    """
    genome_path = Path(genome_dir)
    genome_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"Setting up genome in: {genome_path}")

    # Download genome
    genome_fa = genome_path / "genome.fa"
    genome_gtf = genome_path / "genome.gtf"

    if genome_fa.exists() and skip_download:
        click.echo("✓ Genome FASTA already exists, skipping download")
    else:
        click.echo("Downloading Drosophila melanogaster genome (BDGP6.32)...")
        download_genome(genome_path)

    if genome_gtf.exists() and skip_download:
        click.echo("✓ Genome annotation already exists, skipping download")
    else:
        click.echo("Downloading genome annotation (Ensembl 106)...")
        download_annotation(genome_path)

    # Build index
    if build_index:
        click.echo("Building HISAT2 index...")
        build_hisat2_index(genome_path)

    click.secho("✓ Setup completed successfully!", fg="green", bold=True)


def download_genome(genome_dir: Path):
    """Download D. melanogaster genome from Ensembl"""
    base_url = (
        "https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/"
    )
    chromosomes = ["2L", "2R", "3L", "3R", "4", "X", "Y", "mitochondrion_genome"]

    genome_files = []
    for chrom in chromosomes:
        filename = (
            f"Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.{chrom}.fa.gz"
        )
        url = base_url + filename
        output = genome_dir / f"genome.{chrom}.fa.gz"

        click.echo(f"  Downloading chromosome {chrom}...")
        subprocess.run(["wget", "-q", url, "-O", str(output)], check=True)
        genome_files.append(str(output))

    # Concatenate all chromosomes
    click.echo("  Concatenating chromosomes...")
    genome_fa = genome_dir / "genome.fa"
    with open(genome_fa, "w") as outfile:
        for gz_file in genome_files:
            subprocess.run(["zcat", gz_file], stdout=outfile, check=True)
            Path(gz_file).unlink()  # Remove gz file

    click.echo("✓ Genome download complete")


def download_annotation(genome_dir: Path):
    """Download D. melanogaster annotation from Ensembl"""
    url = "https://ftp.ensembl.org/pub/release-106/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.32.106.chr.gtf.gz"
    output_gz = genome_dir / "genome.gtf.gz"
    output_gtf = genome_dir / "genome.gtf"

    click.echo("  Downloading annotation...")
    subprocess.run(["wget", "-q", url, "-O", str(output_gz)], check=True)

    click.echo("  Decompressing...")
    subprocess.run(["gzip", "-d", "--force", str(output_gz)], check=True)

    click.echo("✓ Annotation download complete")


def build_hisat2_index(genome_dir: Path):
    """Build HISAT2 index from genome FASTA"""
    genome_fa = genome_dir / "genome.fa"
    index_base = genome_dir / "genome.idx"

    if not genome_fa.exists():
        click.secho("✗ Genome FASTA not found!", fg="red")
        return

    # Extract splice sites
    genome_gtf = genome_dir / "genome.gtf"
    splice_sites = genome_dir / "genome.ss"

    if genome_gtf.exists():
        click.echo("  Extracting splice sites...")
        cmd = f"hisat2_extract_splice_sites.py {genome_gtf} > {splice_sites}"
        subprocess.run(cmd, shell=True, check=True)

    # Build index
    click.echo("  Building HISAT2 index (this may take a while)...")
    cmd = ["hisat2-build", "-p", "8", str(genome_fa), str(index_base)]

    log_file = genome_dir / "idx.out.txt"
    err_file = genome_dir / "idx.err.txt"

    with open(log_file, "w") as log_out, open(err_file, "w") as log_err:
        subprocess.run(cmd, stdout=log_out, stderr=log_err, check=True)

    click.echo("✓ HISAT2 index build complete")


@main.command("config")
@click.option(
    "--template",
    "-t",
    is_flag=True,
    help="Generate a template configuration file",
)
@click.option(
    "--output",
    "-o",
    default="config.yaml",
    type=click.Path(),
    help="Output path for configuration file",
)
@click.option(
    "--full",
    "-f",
    is_flag=True,
    help="Generate full example with all options and documentation",
)
def config_cmd(template, output, full):
    """
    Generate pipeline configuration files.

    By default, creates a minimal template with all options commented out.
    Use --full to generate a comprehensive example with documentation.
    """
    if not template:
        click.echo("Use --template to generate a configuration file")
        click.echo("  flync config --template               # Minimal template")
        click.echo("  flync config --template --full        # Full example with docs")
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if full:
        # Copy the full example file
        import shutil
        import flync

        flync_dir = Path(flync.__file__).parent
        full_example = flync_dir.parent.parent / "config_example_full.yaml"

        if full_example.exists():
            shutil.copy(full_example, output_path)
            click.secho(
                f"✓ Full configuration example written to: {output_path}", fg="green"
            )
        else:
            click.secho(
                f"✗ Full example template not found at {full_example}", fg="red"
            )
            sys.exit(1)
    else:
        # Generate minimal template with all options commented out
        minimal_template = """# FLYNC Pipeline Configuration
# Uncomment and modify the options you need

# ==============================================================================
# SAMPLE SPECIFICATION (Required - choose one mode)
# ==============================================================================

# Mode 1: Auto-detect from FASTQ directory (recommended)
samples: null
fastq_dir: "/path/to/fastq"
fastq_paired: false  # true for paired-end, false for single-end

# Mode 2: Plain text sample list
# samples: "samples.txt"
# fastq_dir: "/path/to/fastq"

# Mode 3: CSV metadata (required for DGE - MUST have header row)
# samples: "metadata.csv"  # Must have headers: sample_id,condition
# fastq_dir: "/path/to/fastq"

# ==============================================================================
# REFERENCE GENOME (Required)
# ==============================================================================

genome: "genome/genome.fa"
annotation: "genome/genome.gtf"
hisat_index: "genome/genome.idx"
# splice_sites: "genome/genome.ss"  # Optional, auto-generated

# ==============================================================================
# OUTPUT AND RESOURCES (Required)
# ==============================================================================

output_dir: "results"
threads: 8

# ==============================================================================
# TOOL PARAMETERS (Optional)
# ==============================================================================

# params:
#   hisat2: "-p 8 --dta --dta-cufflinks"
#   stringtie_assemble: "-p 8"
#   stringtie_merge: ""
#   stringtie_quantify: "-eB"
#   download_threads: 4

# ==============================================================================
# MACHINE LEARNING (Required for 'flync run-all')
# ==============================================================================

ml_reference_genome: "genome/genome.fa"
ml_output_file: "results/lncrna_predictions.csv"

# Optional ML parameters
# ml_bwq_config: "config/bwq_config.yaml"
# ml_model: "path/to/custom_model.pkl"
# ml_cache_dir: "/path/to/cache"
# ml_gtf: "results/assemblies/merged-new-transcripts.gtf"
# ml_threads: 8

# ==============================================================================
# NOTES
# ==============================================================================

# For full documentation, see: flync config --template --full
# Or visit: https://github.com/homemlab/flync
"""

        with open(output_path, "w") as f:
            f.write(minimal_template)

        click.secho(
            f"✓ Minimal configuration template written to: {output_path}", fg="green"
        )
        click.echo(f"\nNext steps:")
        click.echo(f"  1. Edit {output_path} with your paths and settings")
        click.echo(f"  2. Run: flync run-all --configfile {output_path}")
        click.echo(f"\nFor full example with all options:")
        click.echo(f"  flync config --template --full -o config_full.yaml")


if __name__ == "__main__":
    main()
