![FLYNC logo](logo.jpeg)

# FLYNC - FLY Non-Coding gene discovery & classification

## TL;DR (Quick Start)

Install (conda recommended):
```bash
conda create -n flync -c bioconda -c conda-forge flync
conda activate flync
```

Setup genome (dm6):
```bash
flync setup --genome-dir genome
```

Generate config template and edit paths:
```bash
flync config --template --output config.yaml
# Set: samples: null and fastq_dir: "/path/to/fastq"
```

Run bioinformatics (auto-detect FASTQs):
```bash
flync run-bio -c config.yaml -j 8
```

Predict lncRNAs (novel transcripts):
```bash
flync run-ml -g results/assemblies/merged-new-transcripts.gtf \
  -o results/lncrna_predictions.csv -r genome/genome.fa -t 8
```

All-in-one:
```bash
flync run-all -c config.yaml -j 8
```

Essential outputs:
- results/assemblies/merged-new-transcripts.gtf (novel)
- results/lncrna_predictions.csv (predictions)
- results/dge/... (if metadata CSV with condition provided)

Need help? Run:
```bash
flync run-bio -c config.yaml --dry-run
```

## Minimal Conceptual Overview

1. Input: FASTQs (local) or SRA IDs (via metadata CSV).
2. Snakemake workflow builds transcriptome and isolates novel transcripts.
3. Feature engine converts GTF + genome into a model-ready feature matrix.
4. Pre-trained model classifies lncRNA vs protein-coding; outputs probabilities.
5. Optional differential expression if conditions provided.

## When to Use Which Command

- run-bio: You only need assemblies.
- run-ml: You have a GTF and want predictions.
- run-all: End-to-end (recommended for new users).
- setup: Prepare genome and indices once.
- config: Generate or validate config.yaml.

## Common Pitfalls (Fast Answers)

| Issue | Fix |
|-------|-----|
| samples: null fails | Ensure fastq_dir is set |
| Snakefile not found | pip install -e . |
| Missing genome index | Re-run flync setup |
| All predictions identical | Check feature extraction logs |
| DGE missing | Ensure metadata CSV has header + condition column |

Full detailed documentation continues below.

---

## Table of Contents

- [TL;DR (Quick Start)](#tldr-quick-start)
- [Minimal Conceptual Overview](#minimal-conceptual-overview)
- [When to Use Which Command](#when-to-use-which-command)
- [Common Pitfalls (Fast Answers)](#common-pitfalls-fast-answers)
- [Pipeline Overview](#pipeline-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [1. Setup Reference Genome](#1-setup-reference-genome)
  - [2. Configure Pipeline](#2-configure-pipeline)
  - [Sample Specification (3 Modes)](#sample-specification-3-modes)
  - [3. Run Bioinformatics Pipeline](#3-run-bioinformatics-pipeline)
  - [4. Run ML Prediction](#4-run-ml-prediction)
  - [5. Run Complete Pipeline (Recommended)](#5-run-complete-pipeline-recommended)
  - [6. Differential Gene Expression (DGE)](#6-differential-gene-expression-dge)
  - [7. Python API Usage](#7-python-api-usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Pipeline Overview

FLYNC executes a complete lncRNA discovery workflow with three main execution modes:

### Phase 1: Bioinformatics Pipeline (`flync run-bio`)
Run only the RNA-seq processing and assembly:
1. **Read Mapping** - Align RNA-seq reads to reference genome using HISAT2
2. **Transcriptome Assembly** - Reconstruct transcripts per sample with StringTie
3. **Assembly Merging** - Create unified transcriptome with gffcompare
4. **Novel Transcript Extraction** - Identify transcripts not in reference annotation
5. **Quantification** - Calculate expression levels per transcript
6. **DGE Analysis** (optional) - Differential expression with Ballgown when metadata.csv provided

### Phase 2: ML Prediction (`flync run-ml`)
Run only the machine learning classification:
1. **Feature Extraction** - Extract multi-modal genomic features
2. **Feature Cleaning** - Standardize and prepare features for ML
3. **ML Classification** - Predict lncRNA candidates using trained EBM model
4. **Confidence Scoring** - Provide prediction probabilities and confidence scores

### Complete Pipeline (`flync run-all`)
Run the entire workflow end-to-end with a single command

---

## Key Features

**Complete End-to-End Pipeline** - Single `flync run-all` command for full workflow  
**Unified Environment** - All dependencies managed in single `environment.yml`  
**Differential Expression** - Integrated Ballgown DGE analysis for condition comparisons  
**Public Python API** - Use FLYNC programmatically in custom workflows  
**Flexible Input Modes** - Auto-detect samples from FASTQ directory or use sample lists  
**Snakemake Orchestration** - Robust workflow management with automatic parallelization  
**Comprehensive Features** - 100+ genomic features from multiple data sources  
**Intelligent Caching** - Downloads and caches remote genomic tracks automatically  
**Production-Ready Models** - Pre-trained EBM classifier with high accuracy  
**Multi-Stage Docker** - Runtime and pre-warmed images for flexible deployment  
**Python 3.11** - Modern Python codebase with type hints and comprehensive documentation  

---

## Installation

### Overview

Choose an installation method based on what you need:

| Use Case | Recommended Method | Command |
|----------|-------------------|---------|
| Full pipeline (alignment + assembly + ML) | Conda (base) | `conda create -n flync -c bioconda -c conda-forge flync` |
| Add differential expression (Ballgown) | Conda add-on | `conda install -n flync flync-dge` |
| ML / feature extraction only (no aligners) | pip + extras | `pip install flync[features,ml]` |
| Programmatic Snakemake orchestration (no bio tools) | pip minimal + workflow | `pip install flync[workflow]` |
| Reproducible container execution | Docker (runtime) | `docker pull ghcr.io/homemlab/flync:latest` |
| Faster startup with pre-cached tracks | Docker (prewarmed) | `docker pull ghcr.io/homemlab/flync:latest-prewarmed` |

### Option 1: Conda (Recommended – Full Stack)

```bash
conda create -n flync -c bioconda -c conda-forge flync
conda activate flync
flync --help
```

Add DGE support (Ballgown + R stack):
```bash
conda install -n flync flync-dge  # after base install
```

Or install both at once:
```bash
conda create -n flync -c bioconda -c conda-forge flync flync-dge
```

### Option 2: pip (Python-Only / Lightweight)

Pip will NOT install external bioinformatics binaries (HISAT2, StringTie, samtools, etc.). Use this only for feature extraction or ML inference on an existing GTF.

```bash
python -m venv flync-venv
source flync-venv/bin/activate
pip install --upgrade pip

# Feature extraction + ML
pip install "flync[features,ml]"

# Add Snakemake lightweight orchestration (still no external binaries)
pip install "flync[workflow]"

flync run-ml --help
```

If you attempt `flync run-bio` without the required external tools, FLYNC will explain what is missing and how to install via conda.

### Option 3: Docker

Runtime image (downloads tracks on demand):
```bash
docker pull ghcr.io/homemlab/flync:latest
docker run --rm -v $PWD:/work ghcr.io/homemlab/flync:latest \
  flync --help
```

Prewarmed image (tracks pre-cached):
```bash
docker pull ghcr.io/homemlab/flync:latest-prewarmed
```

### Which Should I Pick?

| Scenario | Choose |
|----------|-------|
| New user, want everything | Conda base (add `flync-dge` if doing DGE) |
| HPC / cluster with module rules | Conda (export env YAML for reproducibility) |
| Notebook exploratory ML only | pip extras (`features,ml`) |
| CI / workflow integration | Docker runtime image |
| Need fastest repeated ML runs | Docker prewarmed image |

### External Tool Summary

The following are ONLY installed automatically via the Conda packages (`flync`, `flync-dge`):
```
hisat2, stringtie, gffcompare, gffread, samtools, bedtools, sra-tools,
R (r-base), bioconductor-ballgown, r-matrixstats, r-ggplot2
```
Pip installations will perform a dependency sanity check and abort `run-bio` if these are missing (unless `--skip-deps-check` is used).

### Development Install (Editable)

```bash
git clone https://github.com/homemlab/flync.git
cd flync
conda env create -f environment.yml
conda activate flync
pip install -e .
```

### Prerequisites

- **Operating System**: Linux (tested on Debian/Ubuntu)
- **Conda/Mamba**: Required for managing dependencies
- **System Requirements**:
  - 8+ GB RAM (16+ GB recommended for large datasets)
  - 20+ GB disk space (genome, indices, and tracks)
  - 4+ CPU cores (8+ recommended)

### Install from Source (Full + Editable)

For development or if you need the latest unreleased features:

```bash
# 1. Clone the repository
git clone https://github.com/homemlab/flync.git
cd flync
git checkout master  # Use the master branch (production)

# 2. Create conda environment with dependencies
conda env create -f environment.yml

# 3. Activate environment
conda activate flync

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
flync --help
```

Docker image details moved above for quick discovery.

---

## Quick Start

**Complete workflow with `run-all` command:**

```bash
# 1. Activate conda environment
conda activate flync

# 2. Download genome and build indices
flync setup --genome-dir genome

# 3. Create configuration file
flync config --template --output config.yaml

# 4. Edit config.yaml with your paths and settings
# See config_example_full.yaml for all available options

# 5. Create metadata.csv with sample information (MUST have header row!)
cat > metadata.csv << EOF
sample_id,condition,replicate
SRR123456,control,1
SRR123457,control,2
SRR123458,treatment,1
SRR123459,treatment,2
EOF

# 6. Update config.yaml to use metadata.csv
# Change: samples: null
# To:     samples: metadata.csv

# 7. Run complete pipeline (bioinformatics + ML + DGE)
flync run-all --configfile config.yaml --cores 8
```

**Alternative: Step-by-step workflow:**

```bash
# Run bioinformatics pipeline only
flync run-bio --configfile config.yaml --cores 8

# Then run ML prediction
flync run-ml \
  --gtf results/assemblies/merged-new-transcripts.gtf \
  --output results/lncrna_predictions.csv \
  --ref-genome genome/genome.fa \
  --threads 8
```

**Python API Usage:**

```python
from flync import run_pipeline
from pathlib import Path

# Run complete pipeline programmatically
result = run_pipeline(
    config_path=Path("config.yaml"),
    cores=8,
    ml_threads=8
)

print(f"Status: {result['status']}")
print(f"Predictions: {result['predictions_file']}")
```

**Output:**
- `results/assemblies/merged.gtf` - Full transcriptome (reference + novel)
- `results/assemblies/merged-new-transcripts.gtf` - Novel transcripts only
- `results/cov/` - Per-sample quantification files
- `results/dge/` - Differential expression analysis (if metadata.csv provided)
  - `transcript_dge_results.csv` - Transcript-level DE results
  - `gene_dge_results.csv` - Gene-level DE results
  - `dge_summary.csv` - Summary statistics
- `results/lncrna_predictions.csv` - lncRNA predictions with confidence scores

---

## Usage Guide

### 1. Setup Reference Genome

Download *Drosophila melanogaster* genome (BDGP6.32/dm6) and build HISAT2 index:

```bash
flync setup --genome-dir genome
```

**What this does:**
- Downloads genome FASTA from Ensembl (release 106)
- Downloads gene annotation GTF
- Builds HISAT2 index (~10 minutes, requires ~4GB RAM)
- Extracts splice sites for splice-aware alignment

**Skip download if files exist:**
```bash
flync setup --genome-dir genome --skip-download
```

### 2. Configure Pipeline

Generate a configuration template:

```bash
flync config --template --output config.yaml
```

**Edit `config.yaml`** with your settings:

```yaml
# Sample specification (3 options - see below)
samples: null                           # Auto-detect from fastq_dir
fastq_dir: "/path/to/fastq/files"      # Directory with FASTQ files
fastq_paired: false                    # true for paired-end, false for single-end

# Reference files (created by 'flync setup')
genome: "genome/genome.fa"
annotation: "genome/genome.gtf"
hisat_index: "genome/genome.idx"
splice_sites: "genome/genome.ss"

# Output and resources
output_dir: "results"
threads: 8

# Tool parameters (optional)
params:
  hisat2: "-p 8 --dta --dta-cufflinks"
  stringtie_assemble: "-p 8"
  stringtie_merge: ""
  stringtie_quantify: "-eB"
  download_threads: 4  # For SRA downloads
```

#### Sample Specification (3 Modes)

**Mode 1: Auto-detect from FASTQ directory (Recommended)**
```yaml
samples: null  # Must be null to enable auto-detection
fastq_dir: "/path/to/fastq"
fastq_paired: false
```

Automatically detects samples from filenames:
- **Paired-end**: `sample1_1.fastq.gz` + `sample1_2.fastq.gz` → detects `sample1`
- **Single-end**: `sample1.fastq.gz` → detects `sample1`

**Mode 2: Plain text list**
```yaml
samples: "samples.txt"
fastq_dir: "/path/to/fastq"  # Optional if using SRA
```

`samples.txt`:
```
sample1
sample2
sample3
```

**Mode 3: CSV with metadata (for differential expression)**
```yaml
samples: "metadata.csv"
fastq_dir: "/path/to/fastq"  # Optional if using SRA
```

`metadata.csv`:
```csv
sample_id,condition,replicate
sample1,control,1
sample2,control,2
sample3,treated,1
```

**⚠️ Important:** When using CSV metadata, the header row with column names (`sample_id`, `condition`) is **required**. Without headers, the DGE analysis will fail. The `sample_id` column is mandatory for sample identification, and the `condition` column is required to enable differential expression analysis.

### 3. Run Bioinformatics Pipeline

Execute the complete RNA-seq workflow:

```bash
flync run-bio --configfile config.yaml --cores 8
```

**What happens:**
1. **Read Mapping**: HISAT2 aligns reads to genome (splice-aware)
2. **Assembly**: StringTie reconstructs transcripts per sample
3. **Merging**: Combines assemblies into unified transcriptome
4. **Comparison**: gffcompare identifies novel vs. known transcripts
5. **Quantification**: StringTie calculates TPM and FPKM values

**Input Modes:**

**A. Local FASTQ files** (set `fastq_dir` in config)
```bash
flync run-bio --configfile config.yaml --cores 8
```

**B. SRA accessions** (omit `fastq_dir`, provide SRA IDs in samples)
```csv
# samples.csv
sample_id,condition,replicate
SRR1234567,control,1
SRR1234568,treated,1
```

SRA files are automatically downloaded using `prefetch` + `fasterq-dump`.

**Useful Options:**
```bash
# Dry run - show what would be executed
flync run-bio -c config.yaml --dry-run

# Unlock after crash
flync run-bio -c config.yaml --unlock

# More cores for faster processing
flync run-bio -c config.yaml --cores 16
```

**Output Structure:**
```
results/
├── data/                           # Alignment files
│   └── {sample}/
│       └── {sample}.sorted.bam
├── assemblies/
│   ├── stringtie/                  # Per-sample assemblies
│   │   └── {sample}.rna.gtf
│   ├── merged.gtf                  # Unified transcriptome
│   ├── merged-new-transcripts.gtf  # Novel transcripts only
│   └── assembled-new-transcripts.fa # Novel transcript sequences
├── gffcompare/
│   └── gffcmp.stats               # Assembly comparison stats
├── cov/                           # Expression quantification
│   └── {sample}/
│       └── {sample}.rna.gtf
└── logs/                          # Per-rule log files
```

### 4. Run ML Prediction

Classify novel transcripts as lncRNA or protein-coding:

```bash
flync run-ml \
  --gtf results/assemblies/merged-new-transcripts.gtf \
  --output results/lncrna_predictions.csv \
  --ref-genome genome/genome.fa \
  --threads 8
```

**Required Arguments:**
- `--gtf`, `-g`: Input GTF file (novel transcripts or full assembly)
- `--output`, `-o`: Output CSV file for predictions
- `--ref-genome`, `-r`: Reference genome FASTA file

- `--output`, `-o`: Output CSV file for predictions
- `--ref-genome`, `-r`: Reference genome FASTA file

**Optional Arguments:**
- `--model`, `-m`: Custom trained model (default: bundled EBM model)
- `--bwq-config`: Custom BigWig track configuration
- `--threads`, `-t`: Number of threads (default: 8)
- `--cache-dir`: Cache directory for downloaded tracks (default: `./bwq_tracks`)
- `--clear-cache`: Clear cache before starting

**What happens:**
1. **Sequence Extraction**: Extracts spliced transcript sequences from GTF
2. **K-mer Profiling**: Calculates 3-12mer frequencies with TF-IDF + SVD
3. **BigWig Query**: Queries 50+ genomic tracks (chromatin, conservation, etc.)
4. **Structure Prediction**: Calculates RNA minimum free energy
5. **Feature Cleaning**: Standardizes features and aligns with model schema
6. **ML Prediction**: Classifies using pre-trained EBM model

**Output Format (`lncrna_predictions.csv`):**
```csv
transcript_id,prediction,confidence,probability_lncrna
MSTRG.1.1,1,0.95,0.95
MSTRG.1.2,0,0.87,0.13
MSTRG.2.1,1,0.89,0.89
```

**Column Descriptions:**
- `transcript_id`: Transcript identifier from GTF
- `prediction`: 1 = lncRNA, 0 = protein-coding
- `confidence`: Model confidence score (0-1)
- `probability_lncrna`: Probability of being lncRNA (0-1)

**Filter high-confidence lncRNAs:**
```bash
# Get lncRNAs with >90% confidence
awk -F',' '$3 > 0.90 && $2 == 1' results/lncrna_predictions.csv > high_conf_lncrnas.csv
```

### 5. Run Complete Pipeline (Recommended)

Execute both bioinformatics and ML prediction with a single command:

```bash
flync run-all --configfile config.yaml --cores 8
```

**Unified Configuration:**

```yaml
# Bioinformatics settings
samples: metadata.csv
genome: genome/genome.fa
annotation: genome/genome.gtf
hisat_index: genome/genome.idx
output_dir: results
threads: 8

# ML settings (required for run-all)
ml_reference_genome: genome/genome.fa
ml_output_file: results/lncrna_predictions.csv
ml_bwq_config: config/bwq_config.yaml  # Optional
ml_cache_dir: /path/to/cache          # Optional
```

**What happens:**
1. Runs bioinformatics pipeline (`flync run-bio`)
2. Automatically detects output GTF (`results/assemblies/merged-new-transcripts.gtf`)
3. Runs ML prediction on novel transcripts
4. Generates DGE analysis if `metadata.csv` has condition column

**Options:**
```bash
# Skip bioinformatics (use existing GTF)
flync run-all -c config.yaml --skip-bio

# Skip ML prediction (only run bioinformatics)
flync run-all -c config.yaml --skip-ml

# Dry run to see what would be executed
flync run-all -c config.yaml --dry-run

# Custom thread allocation
flync run-all -c config.yaml --cores 16 --ml-threads 8
```

### 6. Differential Gene Expression (DGE)

Run DGE analysis using Ballgown when metadata with conditions is provided:

**Requirements:**
- `samples` config key points to a CSV file (not TXT)
- CSV **must have a header row** with column names
- CSV **must contain** `sample_id` column (for sample identification)
- CSV **must contain** `condition` column (for grouping samples in DGE)

**Example metadata.csv:**
```csv
sample_id,condition,replicate
SRR123456,control,1
SRR123457,control,2
SRR123458,treatment,1
SRR123459,treatment,2
```

**⚠️ Critical:** The header row is **not optional**. If you omit it or have a headerless CSV, the DGE analysis will fail with an error about missing the `sample_id` column.

**DGE runs automatically** when using `flync run-bio` or `flync run-all` with metadata CSV.

**Output Files:**
```
results/dge/
├── transcript_dge_results.csv  # Transcript-level differential expression
├── gene_dge_results.csv        # Gene-level differential expression
├── dge_summary.csv             # Analysis summary statistics
├── transcript_ma_plot.png      # MA plot visualization
└── ballgown_dge.log           # Analysis log
```

**DGE Results Format:**
```csv
id,pval,qval,fc,gene_name,gene_id
MSTRG.1.1,0.001,0.01,2.5,gene_A,FBgn0001
MSTRG.1.2,0.05,0.12,1.8,gene_B,FBgn0002
```

**Filter significant transcripts:**
```bash
# Get transcripts with FDR < 0.05
awk -F',' '$3 < 0.05' results/dge/transcript_dge_results.csv > significant_de.csv
```

### 7. Python API Usage

Use FLYNC programmatically in custom workflows:

```python
from flync import run_pipeline, run_bioinformatics, run_ml_prediction
from pathlib import Path

# Run complete pipeline
result = run_pipeline(
    config_path=Path("config.yaml"),
    cores=8,
    ml_threads=8,
    verbose=True
)

if result['status'] == 'success':
    print(f"✓ Pipeline completed!")
    print(f"  Predictions: {result['predictions_file']}")
    print(f"  Output directory: {result['output_dir']}")
```

**Run only bioinformatics:**
```python
from flync import run_bioinformatics

result = run_bioinformatics(
    config_path=Path("config.yaml"),
    cores=16,
    verbose=True
)
```

**Run only ML prediction:**
```python
from flync import run_ml_prediction

result = run_ml_prediction(
    gtf_file=Path("merged.gtf"),
    output_file=Path("predictions.csv"),
    ref_genome=Path("genome.fa"),
    threads=8,
    verbose=True
)

print(f"Predicted {result['n_lncrna']} lncRNAs")
```

**Integration in larger workflows:**
```python
import flync

# Part of a larger analysis pipeline
def analyze_rnaseq_data(sample_dir, output_dir):
    # Run FLYNC
    result = flync.run_pipeline(
        config_path=create_config(sample_dir, output_dir),
        cores=8
    )
    
    # Continue with downstream analyses
    if result['status'] == 'success':
        lncrnas = pd.read_csv(result['predictions_file'])
        perform_enrichment_analysis(lncrnas)
        generate_report(lncrnas, result['output_dir'])
```

---

## Pipeline Architecture

FLYNC follows a modular Python-first architecture with unified CLI:

```
┌─────────────────────────────────────────────────────────────┐
│                   CLI Layer (click)                         │
│  flync run-all | run-bio | run-ml | setup | config          │
│  + Public Python API (flync.run_pipeline)                   │
└──────────────┬────────────────────────┬─────────────────────┘
               │                        │
     ┌─────────▼────────┐    ┌─────────▼──────────┐
     │  Bioinformatics  │    │   ML Prediction    │
     │    (Snakemake)   │    │    (Python)        │
     └─────────┬────────┘    └─────────┬──────────┘
               │                       │
     ┌─────────▼────────┐    ┌─────────▼──────────┐
     │  Workflow Rules  │    │ Feature Extraction │
     │  - mapping.smk   │    │  - feature_wrapper │
     │  - assembly.smk  │    │  - bwq, kmer, mfe  │
     │  - merge.smk     │    │  - cleaning        │
     │  - quantify.smk  │    │                    │
     │  - dge.smk       │    │                    │
     └──────────────────┘    └─────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   ML Predictor      │
                            │  - EBM model        │
                            │  - Schema validator │
                            └─────────────────────┘
```

### Core Components

**1. CLI (`src/flync/cli.py`) & API (`src/flync/api.py`)**
- Single unified command with 5 subcommands: `run-all`, `run-bio`, `run-ml`, `setup`, `config`
- New `run-all` orchestrates complete pipeline end-to-end
- Public Python API for programmatic access
- Custom error handling and helpful messages
- Absolute path resolution for file operations

**2. Workflows (`src/flync/workflows/`)**
- **Snakefile**: Main workflow orchestrator with conditional DGE
- **rules/mapping.smk**: HISAT2 alignment, SRA download, FASTQ symlinking
- **rules/assembly.smk**: StringTie per-sample assembly
- **rules/merge.smk**: StringTie merge + gffcompare
- **rules/quantify.smk**: Expression quantification
- **rules/dge.smk**: Ballgown differential expression
- **scripts/ballgown_dge.R**: R script for Ballgown DGE analysis
- **scripts/predownload_tracks.py**: Docker image track pre-caching

**3. Feature Extraction (`src/flync/features/`)**
- **feature_wrapper.py**: High-level orchestration
- **bwq.py**: BigWig/BigBed track querying
- **kmer.py**: K-mer profiling with TF-IDF and SVD
- **mfe.py**: RNA secondary structure (MFE calculation)
- **feature_cleaning.py**: Data preparation and schema alignment

**4. ML Prediction (`src/flync/ml/`)**
- **predictor.py**: Main prediction interface
- **ebm_predictor.py**: EBM model wrapper
- **schema_validator.py**: Feature schema validation

**5. Utilities (`src/flync/utils/`)**
- **kmer_redux.py**: K-mer transformation utilities
- **progress.py**: Progress bar management

**6. Assets (`src/flync/assets/`)**
- Pre-trained EBM models and scalers
- Model schema definitions

**7. Configuration (`src/flync/config/`)**
- **bwq_config.yaml**: Default BigWig track configuration

---

## Advanced Usage

### Custom BigWig Track Configuration

Create a custom `bwq_config.yaml` to query your own tracks:

```yaml
# List of BigWig/BigBed files to query
- path: /path/to/custom_track.bigWig
  upstream: 1000    # Extend region upstream
  downstream: 1000  # Extend region downstream
  stats:
    - stat: mean
      name: custom_mean
    - stat: max
      name: custom_max
    - stat: coverage
      name: custom_coverage

- path: https://example.com/remote_track.bigBed
  stats:
    - stat: coverage
      name: remote_coverage
    - stat: extract_names
      name: remote_names
      name_field_index: 3  # For BigBed name extraction
```

**Available Statistics:**
- `mean`, `max`, `min`, `sum`: Numerical summaries
- `std`: Standard deviation
- `coverage`: Fraction of region covered by signal
- `extract_names`: Extract names from BigBed entries

Use with ML prediction:
```bash
flync run-ml --gtf input.gtf --output predictions.csv \
  --ref-genome genome.fa --bwq-config custom_bwq_config.yaml
```

### Feature Extraction Only

Extract features without running prediction:

```bash
python src/flync/features/feature_wrapper.py all \
  --gtf annotations.gtf \
  --ref-genome genome.fa \
  --bwq-config config/bwq_config.yaml \
  --k-min 3 --k-max 12 \
  --use-tfidf --use-dim-redux --redux-n-components 1 \
  --output features.parquet
```

### Training Custom Models

**1. Prepare training data:**
```bash
# Split positive and negative samples
python src/flync/optimizer/prepare_data.py \
  --positive-file lncrna_features.parquet \
  --negative-file protein_coding_features.parquet \
  --output-dir datasets/ \
  --train-size 0.7 --val-size 0.15 --test-size 0.15
```

**2. Optimize hyperparameters:**
```bash
python src/flync/optimizer/hyperparameter_optimizer.py \
  --train-data datasets/train.parquet \
  --test-data datasets/test.parquet \
  --holdout-data datasets/holdout.parquet \
  --model-type randomforest \
  --optimization-metrics precision f1 \
  --n-trials 100 \
  --experiment-name "Custom_RF_Model"
```

**3. View results in MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**4. Extract model schema for inference:**
```bash
python src/flync/ml/schema_extractor.py \
  --model-path best_model.pkl \
  --training-data datasets/train.parquet \
  --output-schema model_schema.json
```

### Docker Deployment

**Build custom image:**
```bash
docker build -t my-flync:latest -f Dockerfile .
```

**Run with mounted volumes:**
```bash
docker run --rm \
  -v $PWD/data:/data \
  -v $PWD/genome:/genome \
  -v $PWD/results:/results \
  my-flync:latest \
  flync run-bio -c /data/config.yaml --cores 8
```

**Interactive shell:**
```bash
docker run -it --rm -v $PWD:/work my-flync:latest /bin/bash
```

---

## Troubleshooting

### Installation Issues

**Problem**: `command not found: flync`
```bash
# Solution: Activate conda environment
conda activate flync

# Verify installation
which flync
flync --version
```

**Problem**: `Snakefile not found` when running `flync run-bio`
```bash
# Solution: Reinstall package in editable mode
pip install -e .
```

**Problem**: Missing bioinformatics tools (hisat2, stringtie, etc.)
```bash
# Solution: Recreate conda environment
conda env remove -n flync
conda env create -f environment.yml
conda activate flync
```

### Pipeline Execution Issues

**Problem**: HISAT2 index build fails
```bash
# Check available disk space (needs ~10GB)
df -h

# Check available memory (needs ~4GB)
free -h

# Check logs
cat genome/idx.err.txt
```

**Problem**: SRA download hangs or fails
```bash
# Solution 1: Reduce download threads in config.yaml
params:
  download_threads: 2  # Instead of 4

# Solution 2: Pre-download SRA files manually
prefetch SRR1234567
fasterq-dump SRR1234567 --outdir fastq/
```

**Problem**: Snakemake workflow crashes
```bash
# Unlock working directory
flync run-bio -c config.yaml --unlock

# Check logs for specific rule
tail -f results/logs/hisat2/sample1.log

# Rerun with verbose output
flync run-bio -c config.yaml --cores 8 --dry-run --printshellcmds
```

**Problem**: `samples: null` fails
```bash
# Solution: Must also set fastq_dir in config.yaml
samples: null
fastq_dir: "/path/to/fastq"  # Required for auto-detection
fastq_paired: false
```

### Feature Extraction Issues

**Problem**: Feature extraction fails with "track not accessible"
```bash
# Solution: Check internet connection (tracks downloaded from UCSC/Ensembl)
wget -q --spider http://genome.ucsc.edu
echo $?  # Should be 0

# Clear cache and retry
flync run-ml --gtf input.gtf --clear-cache ...
```

**Problem**: "No sequences available for downstream feature generation"
```bash
# Solution 1: Verify GTF has transcript and exon features
grep -c 'transcript' input.gtf
grep -c 'exon' input.gtf

# Solution 2: Check reference genome is accessible
ls -lh genome/genome.fa
samtools faidx genome/genome.fa  # Build index if missing
```

**Problem**: "kmer_redux utilities not available"
```bash
# Solution: Verify utils module is installed
python -c "from flync.utils import kmer_redux; print('OK')"

# Reinstall if needed
pip install -e .
```

### ML Prediction Issues

**Problem**: "schema mismatch" error during prediction
```bash
# Solution: Feature transformations must match training
# Ensure these flags are set correctly:
flync run-ml --gtf input.gtf --output predictions.csv \
  --ref-genome genome.fa
# (Default model expects: use_tfidf=True, use_dim_redux=True, redux_n_components=1)
```

**Problem**: Predictions all 0 or all 1
```bash
# Solution 1: Check input GTF quality
# Ensure transcripts are complete and have exons

# Solution 2: Verify feature extraction succeeded
# Check for warnings in logs

# Solution 3: Use different model or retrain
flync run-ml --gtf input.gtf --model custom_model.pkl ...
```

**Problem**: Out of memory during feature extraction
```bash
# Solution 1: Reduce threads
flync run-ml --threads 4 ...

# Solution 2: Process in smaller batches
# Split GTF and process separately

# Solution 3: Use sparse k-mer format (automatic with default settings)
```

### Docker Issues

**Problem**: Docker permission denied
```bash
# Solution 1: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Solution 2: Run with sudo
sudo docker run ...
```

**Problem**: Docker container out of disk space
```bash
# Clean up old containers and images
docker system prune -a

# Check disk usage
docker system df
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/homemlab/flync.git
cd flync
git checkout master

# Create development environment
conda env create -f environment.yml
conda activate flync

# Install in development mode
pip install -e .

# Optional: Install development dependencies
pip install pytest black flake8 mypy
```

### Code Style

- **Python**: Follow PEP 8, use Black formatter (line length 100)
- **Type Hints**: Required for public functions
- **Docstrings**: Google style for all modules, classes, functions
- **Imports**: Absolute imports preferred (`from flync.module import Class`)

### Testing

```bash
# Run tests (when implemented)
pytest tests/

# Format code
black src/flync/

# Type checking
mypy src/flync/
```

### Workflow for Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with clear commit messages
4. Ensure code passes style checks and tests
5. Update documentation if needed
6. Submit a pull request to the `master` branch

### Reporting Issues

- Use GitHub Issues: https://github.com/homemlab/flync/issues
- Include:
  - FLYNC version (`flync --version`)
  - Operating system and version
  - Minimal reproducible example
  - Error messages and logs

## License

MIT License - see [LICENSE](LICENSE) file for details.
