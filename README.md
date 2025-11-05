![FLYNC logo](logo.jpeg)

# FLYNC - FLY Non-Coding gene discovery & classification

FLYNC is a complete bioinformatics pipeline for discovering and classifying long non-coding RNAs (lncRNAs) in *Drosophila melanogaster*. It combines RNA-seq processing, comprehensive genomic feature extraction, and machine learning classification to identify novel lncRNA candidates.

**Version**: 1.0.0 (Python-first architecture)  
**Branch**: v2 (production-ready)  
**Last Updated**: November 2025

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Extraction](#feature-extraction)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Migration from v1](#migration-from-v1)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Pipeline Overview

FLYNC executes a complete lncRNA discovery workflow in two main phases:

### Phase 1: Bioinformatics Pipeline (`flync run-bio`)
1. **Read Mapping** - Align RNA-seq reads to reference genome using HISAT2
2. **Transcriptome Assembly** - Reconstruct transcripts per sample with StringTie
3. **Assembly Merging** - Create unified transcriptome with gffcompare
4. **Novel Transcript Extraction** - Identify transcripts not in reference annotation
5. **Quantification** - Calculate expression levels per transcript

### Phase 2: ML Prediction (`flync run-ml`)
1. **Feature Extraction** - Extract multi-modal genomic features:
   - K-mer frequencies (3-12mers) with TF-IDF and SVD dimensionality reduction
   - BigWig track signals (chromatin marks, CAGE-seq, conservation, ChIP-seq)
   - RNA secondary structure (minimum free energy)
   - Transcript characteristics (length, exon count, GC content)
2. **Feature Cleaning** - Standardize and prepare features for ML
3. **ML Classification** - Predict lncRNA candidates using trained EBM model
4. **Confidence Scoring** - Provide prediction probabilities and confidence scores

---

## Key Features

✅ **Unified CLI Interface** - Single `flync` command with intuitive subcommands  
✅ **Flexible Input Modes** - Auto-detect samples from FASTQ directory or use sample lists  
✅ **Snakemake Orchestration** - Robust workflow management with automatic parallelization  
✅ **Comprehensive Features** - 100+ genomic features from multiple data sources  
✅ **Intelligent Caching** - Downloads and caches remote genomic tracks automatically  
✅ **Production-Ready Models** - Pre-trained EBM classifier with high accuracy  
✅ **Docker Support** - Containerized deployment for reproducibility  
✅ **Python-First** - Modern Python codebase (≥3.9) with type hints  

---

## Installation

### Prerequisites

- **Operating System**: Linux (tested on Debian/Ubuntu)
- **Conda/Mamba**: Required for managing dependencies
- **System Requirements**:
  - 8+ GB RAM (16+ GB recommended for large datasets)
  - 20+ GB disk space (genome, indices, and tracks)
  - 4+ CPU cores (8+ recommended)

### Install from Source

```bash
# 1. Clone the repository
git clone https://github.com/homemlab/flync.git
cd flync
git checkout v2  # Use the v2 branch (production)

# 2. Create conda environment (includes all bioinformatics tools)
conda env create -f environment.yml

# 3. Activate environment
conda activate flync

# 4. Install Python package
pip install -e .

# 5. Verify installation
flync --help
```

**What gets installed:**
- **Bioinformatics tools**: HISAT2, StringTie, gffcompare, samtools, bedtools, SRA-tools
- **Python packages**: pandas, scikit-learn, pyBigWig, gffutils, pyfaidx, snakemake, etc.
- **ML frameworks**: interpret (for EBM), optuna, mlflow (for training)

### Docker

Pre-built Docker images provide a complete, portable environment:

```bash
# Standard image (~2GB)
docker pull rfcdsantos/flync:latest

# Image with pre-downloaded genomic tracks (~5GB, faster for predictions)
docker pull rfcdsantos/flync:local-tracks
```

**Run with Docker:**
```bash
# Mount your data directory and run pipeline
docker run --rm -v $PWD:/data rfcdsantos/flync:latest \
  flync run-bio --configfile /data/config.yaml --cores 8
```

---

## Quick Start

**Complete workflow in 5 commands:**

```bash
# 1. Activate conda environment
conda activate flync

# 2. Download genome and build indices
flync setup --genome-dir genome

# 3. Create configuration file
flync config --template --output config.yaml
# Edit config.yaml with your sample information

# 4. Run bioinformatics pipeline
flync run-bio --configfile config.yaml --cores 8

# 5. Predict lncRNAs
flync run-ml \
  --gtf results/assemblies/merged-new-transcripts.gtf \
  --output results/lncrna_predictions.csv \
  --ref-genome genome/genome.fa \
  --threads 8
```

**Output:**
- `results/assemblies/merged.gtf` - Full transcriptome (reference + novel)
- `results/assemblies/merged-new-transcripts.gtf` - Novel transcripts only
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

### Sample Specification (3 Modes)

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

---

## Pipeline Architecture

FLYNC follows a modular Python-first architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Layer (click)                       │
│  flync setup | config | run-bio | run-ml                   │
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
     └──────────────────┘    └─────────┬──────────┘
                                       │
                            ┌──────────▼──────────┐
                            │   ML Predictor      │
                            │  - EBM model        │
                            │  - Schema validator │
                            └─────────────────────┘
```

### Core Components

**1. CLI (`src/flync/cli.py`)**
- Single unified command with 4 subcommands
- Custom error handling and helpful messages
- Absolute path resolution for file operations

**2. Workflows (`src/flync/workflows/`)**
- **Snakefile**: Main workflow orchestrator
- **rules/mapping.smk**: HISAT2 alignment, SRA download, FASTQ symlinking
- **rules/assembly.smk**: StringTie per-sample assembly
- **rules/merge.smk**: StringTie merge + gffcompare
- **rules/quantify.smk**: Expression quantification

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

## Feature Extraction

FLYNC extracts **100+ features** across multiple categories:

### 1. Sequence Features
- **K-mer frequencies**: 3-12mers with TF-IDF weighting
- **Dimensionality reduction**: SVD per k-mer length (grouped approach)
- **GC content**: Percentage of G+C nucleotides
- **Transcript length**: Total length in base pairs
- **Exon count**: Number of exons per transcript

### 2. Genomic Track Features (50+ tracks)

**Chromatin State:**
- ChromHMM annotations (9 states)
- H3K4me3 (promoter marks)
- H3K27ac (active enhancers)

**Transcription:**
- RNA Pol II ChIP-seq
- CAGE-seq (TSS evidence, plus/minus strand)
- EPDnew promoter annotations

**Conservation:**
- PhyloP scores (evolutionary conservation)
- PhastCons scores (conserved elements)

**Regulatory Elements:**
- JASPAR TF binding sites
- ReMap TF binding sites
- Enhancer annotations

**Coding Potential:**
- CPAT scores (Coding Potential Assessment Tool)
- ORF analysis features

### 3. RNA Structure Features
- **Minimum free energy (MFE)**: Thermodynamic stability
- **Secondary structure**: Optional structure prediction

### 4. Expression Features (if samples provided)
- **Coverage statistics**: Mean, max, min, standard deviation
- **Expression levels**: TPM, FPKM values

### Feature Transformation Pipeline

```
Raw Features
    ↓
GTF Parsing (gffutils) → Sequence Extraction (pyfaidx)
    ↓
K-mer Counting → TF-IDF → Grouped SVD (per k-length)
    ↓
BigWig Query (pyBigWig) → Statistics (mean, max, coverage)
    ↓
RNA MFE (RNAfold) → Structure features
    ↓
Feature Aggregation → Merge on transcript_id
    ↓
Feature Cleaning:
  - Drop coordinates, CPAT columns
  - Fill missing values (domain-specific strategies)
  - Sanitize column names
  - Multi-hot encoding (categorical features)
    ↓
Schema Alignment (inference mode):
  - Load model schema
  - Add missing columns (with defaults)
  - Align feature types and order
    ↓
Scaling (StandardScaler or MinMaxScaler)
    ↓
Model-Ready Features
```

**Key Design Decisions:**

1. **GTF-Centric Workflow**: Always uses GTF as primary input for accurate splice junction handling
2. **Grouped SVD**: SVD applied per k-mer length (3-mers, 4-mers, etc.) for balanced representation
3. **Persistent Caching**: Remote tracks cached in `bwq_persistent_cache/` with URL hashing
4. **Schema Validation**: Inference features must match training schema exactly
5. **Missing Value Strategies**: Domain-specific imputation (signal→0, structure→median/0)

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

## Project Structure

```
flync/
├── README.md                  # This file
├── environment.yml            # Conda environment specification
├── pyproject.toml            # Python package metadata
├── Dockerfile                # Docker image definition
├── config.yaml               # Pipeline configuration (generated)
│
├── src/flync/                # Main Python package
│   ├── __init__.py
│   ├── cli.py                # Command-line interface (Click)
│   │
│   ├── workflows/            # Snakemake workflows
│   │   ├── Snakefile         # Main workflow
│   │   └── rules/            # Workflow rules
│   │       ├── mapping.smk   # Read alignment
│   │       ├── assembly.smk  # Transcriptome assembly
│   │       ├── merge.smk     # Assembly merging
│   │       └── quantify.smk  # Expression quantification
│   │
│   ├── features/             # Feature extraction modules
│   │   ├── README.md         # Feature extraction documentation
│   │   ├── feature_wrapper.py    # High-level orchestrator
│   │   ├── bwq.py            # BigWig query module
│   │   ├── kmer.py           # K-mer profiling
│   │   ├── mfe.py            # RNA structure prediction
│   │   └── feature_cleaning.py   # Data preparation
│   │
│   ├── ml/                   # Machine learning modules
│   │   ├── predictor.py      # Main prediction interface
│   │   ├── ebm_predictor.py  # EBM model wrapper
│   │   └── schema_validator.py   # Feature validation
│   │
│   ├── optimizer/            # Model training and optimization
│   │   ├── README.md         # Optimizer documentation
│   │   ├── hyperparameter_optimizer.py   # Optuna optimization
│   │   └── batch_optimization.py         # Batch training
│   │
│   ├── utils/                # Utility modules
│   │   ├── kmer_redux.py     # K-mer transformations
│   │   └── progress.py       # Progress tracking
│   │
│   ├── assets/               # Bundled models and data
│   │   ├── flync_ebm_model.pkl           # Pre-trained EBM model
│   │   ├── flync_ebm_scaler.pkl          # Feature scaler
│   │   └── flync_ebm_model_schema.json   # Model feature schema
│   │
│   └── config/               # Default configurations
│       └── bwq_config.yaml   # Default BigWig tracks
│
├── genome/                   # Reference genome (generated by 'flync setup')
│   ├── genome.fa             # Reference FASTA
│   ├── genome.gtf            # Reference annotation
│   ├── genome.idx.*.ht2      # HISAT2 index files
│   └── genome.ss             # Splice sites
│
├── test/                     # Test configurations and data
│   ├── config.yaml
│   ├── samples.txt
│   └── metadata.csv
│
├── results/                  # Pipeline outputs (generated)
│   ├── data/                 # Alignment files (BAM)
│   ├── assemblies/           # Transcriptome assemblies (GTF)
│   ├── gffcompare/          # Comparison results
│   ├── cov/                 # Expression quantification
│   └── logs/                # Per-rule log files
│
├── bwq_tracks/              # Cached genomic tracks (generated)
│   ├── bwq_persistent_cache/    # Downloaded BigWig/BigBed files
│   └── gffutils_cache/          # GTF databases
│
└── static/                  # Legacy CPAT files (for reference only)
    ├── fly_Hexamer.tsv
    ├── fly_cutoff.txt
    └── Fly_logitModel.RData
```

### Key Directories

- **`src/flync/`**: Main Python package (all active code)
- **`genome/`**: Reference files (created by `flync setup`)
- **`results/`**: Pipeline outputs (created by `flync run-bio`)
- **`bwq_tracks/`**: Cached genomic tracks (created by `flync run-ml`)
- **`test/`**: Example configurations for testing
- **`static/`**: Legacy files (not used by current version)

### Files to Ignore

The following are not part of the active pipeline:
- `deprecated_v1_*/` - Backed up v1 bash scripts
- `*_out/`, `nb_out/` - Test outputs
- `.github/`, `.vscode/` - Development tools
- `README.old.md` - Outdated documentation
- `nohup.out`, `*.log` - Runtime logs

---

## Migration from v1

If upgrading from the bash-based v1 pipeline:

### Major Changes

1. **Single CLI Command**: All functionality through `flync` command (not separate scripts)
2. **Python Package**: Must install with `pip install -e .`
3. **Snakemake Workflow**: Replaces bash scripts for bioinformatics pipeline
4. **YAML Configuration**: Replaces command-line arguments
5. **Unified Environment**: Single conda environment (not multiple `env/` folders)

### Command Mapping

| v1 Command | v2 Command |
|------------|------------|
| `./flync sra <accessions>` | `flync run-bio -c config.yaml` |
| `./get-genome.sh` | `flync setup --genome-dir genome` |
| `./tux2map.sh` | `flync run-bio -c config.yaml` |
| `./predict.py <args>` | `flync run-ml -g input.gtf -o output.csv` |
| `./feature-table.py` | `python src/flync/features/feature_wrapper.py` |

### Migration Steps

1. **Backup old installation:**
   ```bash
   # Move v1 scripts to backup folder (already done if you see deprecated_v1_*)
   ```

2. **Remove old environments:**
   ```bash
   # v1 used multiple environments in env/
   rm -rf env/
   conda env remove -n flync_old  # If you had custom env names
   ```

3. **Install v2:**
   ```bash
   conda env create -f environment.yml
   conda activate flync
   pip install -e .
   ```

4. **Update configurations:**
   - Replace command-line args with `config.yaml`
   - Update sample file formats (CSV with headers)
   - Update paths to genome files

5. **Test installation:**
   ```bash
   flync --help
   flync setup --genome-dir genome --skip-download
   ```

See `MIGRATION_GUIDE.md` for detailed migration instructions.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/homemlab/flync.git
cd flync
git checkout v2

# Create development environment with optional dev packages
conda env create -f environment.yml
conda activate flync
pip install -e ".[dev]"  # Includes pytest, black, flake8, mypy
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
6. Submit a pull request to the `v2` branch

### Reporting Issues

- Use GitHub Issues: https://github.com/homemlab/flync/issues
- Include:
  - FLYNC version (`flync --version`)
  - Operating system and version
  - Minimal reproducible example
  - Error messages and logs

---

## Citation

If you use FLYNC in your research, please cite:

```bibtex
@software{flync2024,
  title={FLYNC: lncRNA discovery and classification pipeline for Drosophila melanogaster},
  author={Santos, R. and Contributors},
  year={2024},
  version={1.0.0},
  url={https://github.com/homemlab/flync},
  note={Python-first bioinformatics pipeline}
}
```

### Related Publications

- **EBM (Explainable Boosting Machine)**: Nori, H., et al. (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223.
- **HISAT2**: Kim, D., et al. (2019). Graph-based genome alignment and genotyping with HISAT2 and HISAT-genotype. Nature Biotechnology, 37(8), 907-915.
- **StringTie**: Pertea, M., et al. (2015). StringTie enables improved reconstruction of a transcriptome from RNA-seq reads. Nature Biotechnology, 33(3), 290-295.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Support

- **Documentation**: https://github.com/homemlab/flync/wiki
- **Issues**: https://github.com/homemlab/flync/issues
- **Discussions**: https://github.com/homemlab/flync/discussions

---

## Acknowledgments

- **Ensembl** for providing reference genomes and annotations
- **UCSC Genome Browser** for genomic track data
- **FlyBase** for *Drosophila* genome resources
- **NONCODE** and **lncRNAdb** for lncRNA annotations
- Open-source bioinformatics community

---

**Version**: 1.0.0  
**Branch**: v2 (production-ready)  
**Last Updated**: November 2025  
**Maintainers**: FLYNC Contributors
