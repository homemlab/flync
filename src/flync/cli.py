"""Modern CLI interface for FLYNC using Typer."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import PipelineConfig, create_default_config, load_config
from .utils import setup_logging
from .utils.dependencies import DependencyManager

app = typer.Typer(
    name="flync",
    help="FLYNC - FLY Non-Coding gene discovery & classification",
    add_completion=False,
)
console = Console()


def version_callback(value: bool):
    """Show version information."""
    if value:
        console.print(f"FLYNC version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, help="Show version information"
    ),
):
    """
    FLYNC - FLY Non-Coding gene discovery & classification
    
    A modern bioinformatics pipeline for discovering and classifying non-coding genes
    in Drosophila transcriptomic data using machine learning.
    """
    pass


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Configuration file (YAML format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from the last successful step"
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be executed without running"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
    skip_deps_check: bool = typer.Option(
        False,
        "--skip-deps",
        help="Skip dependency checking (not recommended)"
    ),
):
    """Run the full FLYNC pipeline using a configuration file."""
    
    # Check dependencies first (unless skipped)
    if not skip_deps_check:
        console.print("🔍 Checking system dependencies...")
        dep_manager = DependencyManager()
        if not dep_manager.is_ready_for_pipeline():
            console.print("❌ [red]Missing required dependencies. Run 'flync check-deps' for details.[/red]")
            console.print("💡 [dim]Use --skip-deps to bypass this check (not recommended)[/dim]")
            raise typer.Exit(1)
        console.print("✅ [green]All required dependencies found[/green]\n")
    
    try:
        # Load configuration
        pipeline_config = load_config(config)
        
        # Override config with CLI options
        if resume:
            pipeline_config.resume = True
        if dry_run:
            pipeline_config.dry_run = True
        
        # Set up logging
        log_level = "DEBUG" if verbose else pipeline_config.output.log_level
        log_file = pipeline_config.output.output_dir / "flync.log"
        setup_logging(log_level=log_level, log_file=log_file)
        
        # Show configuration summary
        show_config_summary(pipeline_config)
        
        if dry_run:
            console.print("[yellow]Dry run mode - no commands will be executed[/yellow]")
            return
        
        # Run pipeline
        from .workflows.main import run_pipeline
        success = run_pipeline(pipeline_config)
        
        if success:
            console.print("[green]✓ Pipeline completed successfully![/green]")
        else:
            console.print("[red]✗ Pipeline failed![/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def sra(
    sra_list: Path = typer.Option(
        ...,
        "--list", "-l",
        help="File with list of SRA accession numbers",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory for results"
    ),
    metadata: Optional[Path] = typer.Option(
        None,
        "--metadata", "-m",
        help="Metadata CSV file for differential expression analysis",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    threads: int = typer.Option(
        2,
        "--threads", "-t",
        help="Number of threads to use",
        min=1,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
):
    """Run pipeline using SRA accession numbers."""
    
    # Create configuration from CLI arguments
    config_data = {
        "threads": threads,
        "input": {
            "sra_list": str(sra_list),
            "metadata": str(metadata) if metadata else None,
        },
        "output": {
            "output_dir": str(output_dir),
            "log_level": "DEBUG" if verbose else "INFO",
        }
    }
      # Create temporary config file
    temp_config = output_dir / "flync_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_config = PipelineConfig(**config_data)
    
    # Set up logging and run pipeline directly
    log_level = "DEBUG" if verbose else pipeline_config.output.log_level
    log_file = pipeline_config.output.output_dir / "flync.log"
    setup_logging(log_level=log_level, log_file=log_file)
    
    # Show configuration summary
    show_config_summary(pipeline_config)
    
    # Run pipeline
    from .workflows.main_workflow import run_pipeline
    success = run_pipeline(pipeline_config)
    
    if success:
        console.print("[green]✓ Pipeline completed successfully![/green]")
    else:
        console.print("[red]✗ Pipeline failed![/red]")
        raise typer.Exit(1)


@app.command()
def fastq(
    fastq_dir: Path = typer.Option(
        ...,
        "--fastq", "-f",
        help="Directory containing FASTQ files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory for results"
    ),
    paired_end: bool = typer.Option(
        False,
        "--paired", "-p",
        help="Reads are paired-end"
    ),
    metadata: Optional[Path] = typer.Option(
        None,
        "--metadata", "-m",
        help="Metadata CSV file for differential expression analysis",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    threads: int = typer.Option(
        2,
        "--threads", "-t",
        help="Number of threads to use",
        min=1,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
):
    """Run pipeline using local FASTQ files."""
    
    # Create configuration from CLI arguments
    config_data = {
        "threads": threads,
        "input": {
            "fastq_dir": str(fastq_dir),
            "paired_end": paired_end,
            "metadata": str(metadata) if metadata else None,
        },
        "output": {
            "output_dir": str(output_dir),
            "log_level": "DEBUG" if verbose else "INFO",
        }
    }    
    # Create and run pipeline directly for FASTQ mode
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_config = PipelineConfig(**config_data)
    
    # Set up logging
    log_level = "DEBUG" if verbose else pipeline_config.output.log_level
    log_file = pipeline_config.output.output_dir / "flync.log"
    setup_logging(log_level=log_level, log_file=log_file)
    
    # Show configuration summary
    show_config_summary(pipeline_config)
    
    # Run pipeline
    from .workflows.main_workflow import run_pipeline
    success = run_pipeline(pipeline_config)
    
    if success:
        console.print("[green]✓ Pipeline completed successfully![/green]")
    else:
        console.print("[red]✗ Pipeline failed![/red]")
        raise typer.Exit(1)


@app.command()
def predict(
    bed_file: Path = typer.Option(
        ...,
        "--bed", "-b",
        help="BED file with genomic regions to classify",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        help="Output directory for results"
    ),
    model_path: Optional[Path] = typer.Option(
        None,
        "--model", "-m",
        help="Path to trained ML model (uses default if not specified)",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    threads: int = typer.Option(
        2,
        "--threads", "-t",
        help="Number of threads to use",
        min=1,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable verbose logging"
    ),
):
    """Run ML classification on a BED file (feature extraction + prediction only)."""
    
    console.print("[yellow]Feature: ML prediction only - Coming in future release[/yellow]")
    # TODO: Implement prediction-only mode


@app.command()
def init_config(
    output_file: Path = typer.Option(
        "flync_config.yaml",
        "--output", "-o",
        help="Output configuration file path"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Overwrite existing configuration file"
    ),
):
    """Create a default configuration file."""
    
    if output_file.exists() and not force:
        console.print(f"[red]Configuration file already exists: {output_file}[/red]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)
    
    create_default_config(output_file)
    console.print(f"[green]✓ Created default configuration: {output_file}[/green]")
    console.print("\nEdit the configuration file and run:")
    console.print(f"[cyan]flync run --config {output_file}[/cyan]")


@app.command()
def check_deps():
    """Check system dependencies and display installation instructions."""
    console.print("\n🔍 Checking FLYNC system dependencies...\n")
    
    dep_manager = DependencyManager()
    results = dep_manager.check_all_dependencies()
    dep_manager.display_dependency_status(results)
    
    # Show installation commands if tools are missing
    if results["summary"]["required_missing"] > 0 or results["summary"]["optional_missing"] > 0:
        console.print("\n📋 Quick Installation Commands:")
        
        # Conda installation
        install_commands = dep_manager.generate_install_commands(results, "conda")
        console.print("\n[bold cyan]Using Conda (Recommended):[/bold cyan]")
        for cmd in install_commands:
            console.print(f"  {cmd}")
        
        console.print("\n💡 [dim]Tip: Install tools using conda for best compatibility[/dim]")
        console.print("💡 [dim]See SYSTEM_DEPENDENCIES.md for detailed instructions[/dim]")
    
    # Exit with error if required tools are missing
    if results["summary"]["required_missing"] > 0:
        console.print("\n❌ [red]Cannot run FLYNC pipeline without required tools[/red]")
        raise typer.Exit(1)
    else:
        console.print("\n✅ [green]Ready to run FLYNC pipeline![/green]")


@app.command()
def install_deps(
    package_manager: str = typer.Option(
        "conda", 
        "--pm", 
        help="Package manager to use (conda, apt, brew)"
    ),
    dry_run: bool = typer.Option(
        False, 
        "--dry-run", 
        help="Show commands without executing them"
    )
):
    """Generate or execute dependency installation commands."""
    dep_manager = DependencyManager()
    results = dep_manager.check_all_dependencies()
    
    if results["summary"]["required_missing"] == 0 and results["summary"]["optional_missing"] == 0:
        console.print("✅ [green]All dependencies are already installed![/green]")
        return
    
    install_commands = dep_manager.generate_install_commands(results, package_manager)
    
    if dry_run:
        console.print(f"\n📋 Installation commands for {package_manager}:")
        for cmd in install_commands:
            console.print(f"  {cmd}")
        console.print("\n💡 [dim]Run without --dry-run to execute these commands[/dim]")
    else:
        console.print(f"\n🚀 Installing dependencies using {package_manager}...")
        for cmd in install_commands:
            console.print(f"Running: [cyan]{cmd}[/cyan]")
            # Note: Actual execution would require subprocess.run(cmd.split())
            # For safety, we'll just show the commands for now
            console.print("  [dim](Command execution disabled for safety - run manually)[/dim]")
        
        console.print("\n💡 [dim]For safety, commands are not auto-executed. Please run them manually.[/dim]")


def show_config_summary(config: PipelineConfig):
    """Display a summary of the pipeline configuration."""
    
    table = Table(title="FLYNC Pipeline Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Basic settings
    table.add_row("Threads", str(config.threads))
    table.add_row("Memory (GB)", str(config.memory_gb))
    table.add_row("Output Directory", str(config.output.output_dir))
    
    # Input settings
    if config.input.sra_list:
        table.add_row("Input Mode", "SRA")
        table.add_row("SRA List", str(config.input.sra_list))
    elif config.input.fastq_dir:
        table.add_row("Input Mode", "FASTQ")
        table.add_row("FASTQ Directory", str(config.input.fastq_dir))
        table.add_row("Paired-end", str(config.input.paired_end))
    
    if config.input.metadata:
        table.add_row("Metadata", str(config.input.metadata))
    
    # Genome settings
    table.add_row("Genome Species", config.genome.species)
    table.add_row("Genome Release", config.genome.release)
    
    console.print(table)
    console.print()


if __name__ == "__main__":
    app()
