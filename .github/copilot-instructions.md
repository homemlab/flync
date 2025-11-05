# FLYNC AI Agent Instructions

## Project Overview

FLYNC is a **bioinformatics pipeline** for discovering long non-coding RNAs (lncRNAs) in *Drosophila melanogaster*. It combines RNA-seq processing (HISAT2 → StringTie → gffcompare) with machine learning classification (Random Forest/EBM) using genomic features (k-mers, chromatin marks, conservation, RNA structure).

**Recently refactored** (Nov 2025) from bash scripts to Python-first architecture with Snakemake workflow orchestration. Legacy code lives in `deprecated_v1_*` folders.

## Architecture & Core Components

### 1. CLI Entry Point (`src/flync/cli.py`)
- **Single unified command**: `flync` with 4 subcommands
- Uses Click framework with custom `FlyncGroup` for better error messages
- Commands: `setup`, `config`, `run-bio`, `run-ml`
- Always returns absolute paths for file operations

### 2. Bioinformatics Pipeline (`src/flync/workflows/`)
**Managed by Snakemake** (not bash scripts). Workflow structure:
```
Snakefile (main) → rules/mapping.smk → rules/assembly.smk → rules/merge.smk → rules/quantify.smk
```

**Critical workflow patterns:**
- **Config loading**: NO hardcoded `configfile:` directive in Snakefile. Always use `--configfile` CLI arg.
- **Sample auto-detection**: If `samples: null` in config + `fastq_dir` provided, auto-detect from FASTQ filenames using regex patterns (`*_1.fastq.gz`, `*_2.fastq.gz`, `*.fastq.gz`)
- **Dual input modes**: 
  - SRA mode: Download from NCBI using `prefetch` + `fasterq-dump` (requires SRA accessions in samples file)
  - Local FASTQ mode: Symlink files from `fastq_dir` (detected via `config.get("fastq_dir", None) is not None`)
- **Output paths**: Use `OUTPUT_DIR / "subdir/file"` (Path objects), not string concatenation
- **Rule conditions**: See `USE_LOCAL_FASTQ` flag in `mapping.smk` for conditional rule execution

### 3. Feature Extraction Pipeline (`src/flync/features/`)
**Two-stage process** orchestrated by `FeatureWrapper` class:

**Stage 1: Raw feature extraction**
- **GTF-centric workflow**: Always prefer GTF as primary input for accurate splice junctions
- Sequence extraction: `gffutils` for exon concatenation → `pyfaidx` for sequence retrieval → strand-aware reverse complement
- Feature types: BigWig queries (`bwq.py`), k-mers with TF-IDF+SVD (`kmer.py`), RNA MFE (`mfe.py`)
- **Caching strategy**: 
  - Remote tracks cached in `bwq_persistent_cache/` (hashed URLs)
  - gffutils DB cached in `gffutils_cache/` (per GTF file)
  - K-mer artifacts (sparse matrices) optionally saved via `--kmer-sparse-base`

**Stage 2: Feature cleaning** (`feature_cleaning.py`)
- **Training mode**: Splits data (stratified), fits scaler, saves artifacts
- **Inference mode**: Loads model schema, aligns columns, applies saved scaler
- **Column sanitization**: Remove special chars, lowercase, replace spaces with underscores
- **Missing value strategy**: 
  - Signal features (min_, max_, mean_, std_, sum_, cov_) → fill 0
  - K-mer features → fill 0
  - CPAT features → drop entirely
  - Extra ss_ features (except ss_mfe) → drop if >50% missing

### 4. ML Prediction (`src/flync/ml/`)
**High-level API**: `predictor.py::predict_lncrna()` chains extraction → cleaning → prediction
- **Model types**: Random Forest (`.model`), EBM (`.pkl` with separate scaler)
- **Bundled assets**: Models in `src/flync/assets/`, accessed via `flync.__file__` traversal (NOT pkg_resources for reliability)
- **Default BWQ config**: Falls back to `src/flync/config/bwq_config.yaml` if not provided
- **Output format**: CSV with columns: `transcript_id`, `prediction`, `confidence`, `probability_lncrna`

## Critical Developer Workflows

### Running the Pipeline
```bash
# ALWAYS activate conda first (single environment, no more env/ directory juggling)
conda activate flync

# Complete analysis (auto-detect samples from FASTQ directory)
flync setup --genome-dir genome
cat > config.yaml << EOF
samples: null
fastq_dir: "/path/to/fastq"
fastq_paired: false
genome: "genome/genome.fa"
annotation: "genome/genome.gtf"
hisat_index: "genome/genome.idx"
output_dir: "results"
threads: 8
EOF
flync run-bio -c config.yaml -j 8
flync run-ml -g results/assemblies/merged-new-transcripts.gtf -o predictions.csv -r genome/genome.fa
```

### Debug Workflow Issues
```bash
# Snakemake dry-run with shell commands printed
flync run-bio -c config.yaml --dry-run

# Unlock after crash
flync run-bio -c config.yaml --unlock

# Check logs (per-rule logging in logs/ subfolders)
tail -f results/logs/hisat2/{sample}.log
```

### Testing Feature Extraction
```python
from flync.features.feature_wrapper import FeatureWrapper

fw = FeatureWrapper(
    cache_dir="./bwq_tracks",  # Explicit cache location
    threads=8,
    show_progress=True
)

# GTF-based extraction (recommended)
df = fw.run_all(
    gtf_file="merged.gtf",
    ref_genome_path="genome.fa",
    config_file="bwq_config.yaml",
    k_min=3, k_max=12,
    use_dim_redux=True,  # ALWAYS True for inference
    redux_n_components=1,  # Must match training
    use_tfidf=True,  # Must match training
    sparse=False  # False for immediate DataFrame use
)
```

## Project-Specific Conventions

### File Naming & Structure
- **Config files**: Use `config.yaml` (NOT `config.yml`) for consistency
- **Sample formats**: Support 3 modes:
  1. `samples: null` + `fastq_dir` → auto-detect
  2. `samples: "samples.txt"` → plain text list
  3. `samples: "metadata.csv"` → CSV with condition/replicate columns
- **GTF outputs**: 
  - `merged.gtf` = full transcriptome (reference + novel)
  - `merged-new-transcripts.gtf` = novel only (MSTRG IDs)
  - `assembled-new-transcripts.fa` = FASTA of novel transcripts

### Python Coding Patterns
- **Import style**: Use `from flync.module import Class` (NOT relative imports across subpackages)
- **Path handling**: Always `Path()` objects, use `/` operator for joining
- **Type hints**: Required for public functions (uses `typing` module)
- **Click options**: Use single-letter shortcuts (`-c`, `-g`, `-o`, `-r`) for common options
- **Error messages**: Use `click.secho()` with colors: `fg="red"` for errors, `fg="green"` for success

### Snakemake Patterns
- **Input functions**: Use `lambda w: OUTPUT_DIR / f"path/{w.sample}"` for dynamic paths
- **Shell scripts**: Always include `set -e` for error propagation in multi-line shells
- **Log files**: Per-rule logs in `{OUTPUT_DIR}/logs/{rule_name}/{sample}.log`
- **Conditional rules**: Check config keys with `config.get("key", None) is not None` before rule definition

### Feature Extraction Gotchas
- **Schema alignment**: Inference features MUST match training schema exactly (column names, order, types)
- **K-mer transformations**: `use_tfidf=True`, `use_dim_redux=True`, `redux_n_components=1` are NOT optional—they're part of the trained model
- **Grouped SVD**: Default behavior groups by k-mer length (3-mer, 4-mer, etc. get separate SVD). Use `group_kmer_redux_by_length=True` for consistency
- **BWQ stat naming**: Output column names come from `bwq_config.yaml` `name:` field, not auto-generated from track URLs

## Integration Points

### External Dependencies
- **Bioinformatics tools** (managed by conda, NOT pip):
  - HISAT2 (aligner), StringTie (assembler), gffcompare (comparison)
  - samtools (BAM processing), bedtools (interval ops)
  - SRA-tools: `prefetch` + `fasterq-dump` (NOT `fastq-dump`)
- **Genomic data sources**:
  - Ensembl FTP: Reference genome/annotation (dm6/BDGP6.32)
  - UCSC tracks: ChromHMM, CAGE, ChIP-seq (downloaded at runtime, cached)
  - JASPAR/ReMap: TF binding sites (BigBed format)

### Model Training & Optimization
- **MLflow tracking**: Experiments stored in `mlflow.db` (sqlite)
- **Optuna hyperparameter search**: See `src/flync/optimizer/hyperparameter_optimizer.py`
- **Data preparation**: Use `prepare_data.py` for stratified splits before training
- **Model schema**: Extract with `schema_extractor.py` after training for inference alignment

## Common Pitfalls & Solutions

1. **"Snakefile not found"** → Run `pip install -e .` to install package (Snakefile is in package data)
2. **"samples: null" fails** → Must also set `fastq_dir` in config
3. **Schema mismatch in ML** → Ensure `use_tfidf`, `use_dim_redux`, `redux_n_components` match training
4. **SRA download hangs** → Check `download_threads` in config (default: 4). Reduce if rate-limited
5. **HISAT2 index fails** → Needs ~10GB disk + 4GB RAM. Check with `df -h` and `free -h`
6. **Feature wrapper "no sequences"** → Verify GTF has `feature_type="transcript"` and `feature_type="exon"` entries

## Testing Strategy
- **No formal test suite yet** (see deprecation of old tests in `deprecated_v1_*/`)
- **Integration testing**: Use `test/config*.yaml` files as examples
- **Validation**: Run `flync run-bio --dry-run` before actual execution
- **Feature validation**: `feature_cleaning.py --mode validation` checks schema without saving

## Key Files to Reference
- **CLI patterns**: `src/flync/cli.py` (Click setup, error handling)
- **Snakemake conventions**: `src/flync/workflows/Snakefile` (config loading, sample detection)
- **Feature API**: `src/flync/features/README.md` (comprehensive usage guide)
- **Migration notes**: See README.md "Migration from Legacy Version" section

## Version Context
- **Current**: v1.0.0 (Python-first refactoring complete)
- **Branch**: master (production)
- **Python**: ≥3.11 (uses modern typing, dataclasses)
- **Conda**: Required (single `environment.yml`, NOT multiple env/ files)
