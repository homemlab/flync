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
                    "  flync run-ml   - Run ML lncRNA prediction pipeline", err=True
                )
                click.echo(
                    "  flync run-bio  - Run bioinformatics assembly pipeline", err=True
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


@click.group(cls=FlyncGroup)
@click.version_option()
def main():
    """
    FLYNC: lncRNA discovery pipeline for Drosophila melanogaster

    A bioinformatics pipeline for discovering and classifying non-coding genes.
    Combines RNA-seq processing, feature extraction from genomic databases,
    and machine learning prediction.
    """
    pass


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
    default="config/config.yaml",
    type=click.Path(),
    help="Output path for configuration file",
)
def config_cmd(template, output):
    """
    Generate or validate pipeline configuration files.
    """
    if template:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        template_config = {
            "samples": "metadata.csv",
            "genome": "genome/genome.fa",
            "annotation": "genome/genome.gtf",
            "hisat_index": "genome/genome.idx",
            "splice_sites": "genome/genome.ss",
            "output_dir": "results",
            "threads": 8,
            "params": {
                "hisat2": "-p 8 --dta --dta-cufflinks",
                "stringtie_assemble": "-p 8",
                "stringtie_merge": "",
                "stringtie_quantify": "-eB",
                "download_threads": 4,
            },
        }

        with open(output_path, "w") as f:
            yaml.dump(template_config, f, default_flow_style=False, sort_keys=False)

        click.secho(f"✓ Template configuration written to: {output_path}", fg="green")
    else:
        click.echo("Use --template to generate a configuration file")


if __name__ == "__main__":
    main()
