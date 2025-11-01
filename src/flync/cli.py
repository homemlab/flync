#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FLYNC CLI - Command-line interface for the FLYNC pipeline."""

import subprocess
import sys
from pathlib import Path

import click
import yaml

from flync import __version__, __author__, __email__


# COLORS using click.style instead of custom bcolors
def print_header():
    """Print FLYNC header."""
    header = f"""
{click.style('Fly Non-Coding RNA discovery and classification', bold=True, fg='magenta')}
Version: {__version__}
Author: {__author__}
Contact: {__email__}
"""
    click.echo(header)


@click.group()
@click.version_option(version=__version__, prog_name='flync')
@click.pass_context
def cli(ctx):
    """FLYNC - FLY Non-Coding gene discovery & classification.
    
    FLYNC is an end-to-end software pipeline that takes reads from 
    transcriptomic experiments and outputs a curated list of new 
    non-coding genes as classified by a pre-trained Machine-Learning model.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.option(
    '--config', '-c',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default='test/config.yaml',
    help='Configuration file (YAML format) with arguments for running the pipeline. Default = test/config.yaml'
)
@click.pass_context
def run(ctx, config):
    """Run full pipeline using a <config.yaml> file."""
    print_header()
    
    appdir = Path(__file__).resolve().parent.parent.parent
    config_path = Path(config)
    
    with open(config_path, "r") as yaml_file:
        try:
            parsed_yaml = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            click.echo(click.style(f'ERROR: {exc}', fg='red'), err=True)
            sys.exit(1)
    
    if parsed_yaml['fastq_active']:
        # Run the fastq pipeline
        yaml_output = Path(parsed_yaml['output']).absolute().as_posix()
        yaml_fastq_path = Path(parsed_yaml['fastq_path']).absolute().as_posix()
        yaml_fastq_paired = parsed_yaml['fastq_paired']
        yaml_metadata = Path(parsed_yaml['metadata']).absolute().as_posix()
        yaml_threads = str(parsed_yaml['threads'])
        
        cmd = [
            appdir.as_posix() + '/parallel.sh',
            yaml_output,
            yaml_fastq_path,
            yaml_threads,
            appdir.as_posix(),
            yaml_metadata,
            str(yaml_fastq_paired)
        ]
    else:
        # Run the sra pipeline
        yaml_output = Path(parsed_yaml['output']).absolute().as_posix()
        try:
            Path(yaml_output).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            click.echo(click.style(f'ERROR: Permission denied: {yaml_output}', fg='red'), err=True)
            sys.exit(1)
        
        if not Path(parsed_yaml['sra']).is_file():
            click.echo(click.style(f'ERROR: File not found: {parsed_yaml["sra"]}', fg='red'), err=True)
            sys.exit(1)
        
        yaml_sra = Path(parsed_yaml['sra']).absolute().as_posix()
        yaml_metadata = Path(parsed_yaml['metadata']).absolute().as_posix()
        yaml_threads = str(parsed_yaml['threads'])
        
        cmd = [
            appdir.as_posix() + '/parallel.sh',
            yaml_output,
            yaml_sra,
            yaml_threads,
            appdir.as_posix(),
            yaml_metadata
        ]
    
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).communicate()


@cli.command()
@click.option(
    '--list', '-l',
    'sra_list',
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help='MANDATORY. File with list of SRA accession numbers one per line. (usually SRR########)'
)
@click.option(
    '--output', '-o',
    required=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help='MANDATORY. Directory to which the results will be written.'
)
@click.option(
    '--metadata', '-m',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help='(MANDATORY for Diff. Expr. Analysis). Metadata in the .csv format describing the biological condition for each sample.'
)
@click.option(
    '--threads', '-t',
    default=2,
    type=int,
    help='Number of threads to use during execution (default = 2).'
)
@click.pass_context
def sra(ctx, sra_list, output, metadata, threads):
    """Run full pipeline using a <list.txt> of SRA accession numbers."""
    print_header()
    
    appdir = Path(__file__).resolve().parent.parent.parent
    
    cmd = [
        appdir.as_posix() + '/parallel.sh',
        output,
        sra_list,
        str(threads),
        appdir.as_posix(),
        str(metadata) if metadata else 'None'
    ]
    
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).communicate()


@cli.command()
@click.option(
    '--fastq', '-f',
    'fastq_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help='MANDATORY. Directory containing the fastq reads to be analyzed.'
)
@click.option(
    '--output', '-o',
    required=True,
    type=click.Path(file_okay=False, resolve_path=True),
    help='MANDATORY. Directory to which the results will be written.'
)
@click.option(
    '--paired', '-p',
    type=click.Choice(['true', 'false', '1', '0', 'True', 'False'], case_sensitive=False),
    default='false',
    help='Set if provided reads are paired-end or not. Default is false (unpaired reads).'
)
@click.option(
    '--metadata', '-m',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help='(MANDATORY for Diff. Expr. Analysis). Metadata in the .csv format describing the biological condition for each sample.'
)
@click.option(
    '--threads', '-t',
    default=2,
    type=int,
    help='Number of threads to use during execution (default = 2).'
)
@click.pass_context
def fastq(ctx, fastq_dir, output, paired, metadata, threads):
    """Run full pipeline by providing a local <dir> containing the <fastq.gz> files."""
    print_header()
    
    appdir = Path(__file__).resolve().parent.parent.parent
    
    cmd = [
        appdir.as_posix() + '/parallel.sh',
        output,
        fastq_dir,
        str(threads),
        appdir.as_posix(),
        str(metadata) if metadata else 'None',
        paired
    ]
    
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr).communicate()


def main():
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == '__main__':
    main()
