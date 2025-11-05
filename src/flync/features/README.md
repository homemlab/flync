# Feature Pipeline - Genomic Feature Extraction & Preparation

This pipeline provides a comprehensive, two-stage workflow for lncRNA/transcript classification:

1.  **Feature Extraction** (`feature_wrapper.py`) - Extract genomic features from GTF annotations
2.  **Feature Preparation** (`feature_cleaning.py`) - Clean and prepare features for ML models

## Quick Start

Full pipeline from GTF to model-ready training data:

```bash
# Step 1: Extract features from GTF
python feature_wrapper.py all \
    --gtf annotations.gtf \
    --ref-genome genome.fa \
    --bwq-config tracks.yaml \
    --use-dim-redux --redux-n-components 1 \
    --use-tfidf --sparse \
    --output raw_features.parquet

# Step 2: Clean and prepare for training
python feature_cleaning.py \
    --mode training \
    --dataset raw_features.parquet \
    --label-column is_lncrna \
    --output-dir model_data/ \
    --split-suffix "v1" \
    --scaler standard \
    --enable-multi-hot \
    --scaler-path model_data/scaler.pkl
````

-----

## Pipeline Workflow

The full pipeline progresses through two main stages:

### Stage 1: Feature Extraction (`feature_wrapper.py`)

When running the `all` command, feature extraction executes in this order:

1.  **GTF Processing** (if GTF provided):
      * Extract gffutils database (cached)
      * Auto-generate BED file for BWQ (if not provided)
      * Extract properly spliced transcript sequences (if FASTA not provided)
2.  **BigWig Query (BWQ)**:
      * Query signal tracks using BED regions
      * Calculate specified statistics per region
3.  **RNA MFE**:
      * Calculate minimum free energy for sequences
      * Optional secondary structure prediction
4.  **K-mer Profiling**:
      * Extract k-mer frequencies (k\_min to k\_max)
      * Optional TF-IDF transformation
      * Optional SVD dimensionality reduction
5.  **Feature Aggregation**:
      * Merge all features on transcript ID
      * Handle sparse matrices appropriately
      * Calculate transcript lengths
      * Output unified feature table (parquet)

### Stage 2: Feature Cleaning & Preparation (`feature_cleaning.py`)

After feature extraction, `feature_cleaning.py` prepares the raw features for model training or inference:

1.  **Data Cleaning**:
      * Add/validate length features from genomic coordinates
      * Drop non-predictive columns (CPAT, extra SS features, coordinate columns)
      * Handle missing values with domain-specific strategies
      * Remove duplicate transcripts
      * Clean categorical values (e.g., EPDnew promoter rankings)
2.  **Feature Transformations**:
      * Multi-hot encoding for categorical features (optional)
      * Column name sanitization (remove special characters)
      * Feature standardization and sorting
      * Scaling (StandardScaler, MinMaxScaler, or none)
3.  **Schema Enforcement** (inference mode):
      * Load model metadata schema
      * Add missing columns with appropriate defaults
      * Align feature types and order
      * Generate schema difference reports
4.  **Data Splitting** (training mode):
      * Stratified train/validation/test splits
      * Configurable split proportions
      * Maintains class balance across splits
      * Saves fitted scaler for inference
5.  **Output Generation**:
      * Cleaned dataset (parquet)
      * Train/val/test splits (parquet, training mode only)
      * Fitted scaler (pickle, training mode only)
      * Schema diff report (JSON/text, inference mode only)

-----

## Part 1: Feature Extraction (`feature_wrapper.py`)

Unified orchestration layer for comprehensive transcript-level feature extraction.

### 1a. Usage Examples

#### 1\. Complete Pipeline (Recommended)

Extract all features from GTF annotation:

```bash
python feature_wrapper.py all \
    --gtf annotations.gtf \
    --ref-genome genome.fa \
    --bwq-config tracks.yaml \
    --use-dim-redux --redux-n-components 1 \
    --use-tfidf --sparse \
    --output features.parquet
```

#### 2\. With Optional Files

Provide pre-extracted BED and FASTA to skip auto-generation:

```bash
python feature_wrapper.py all \
    --gtf annotations.gtf \
    --bed transcripts.bed \
    --fasta sequences.fasta \
    --ref-genome genome.fa \
    --bwq-config tracks.yaml \
    --output features.parquet
```

#### 3\. Individual Feature Extraction

**BigWig Query only:**

```bash
python feature_wrapper.py bwq \
    --bed regions.bed \
    --bwq-config tracks.yaml \
    --output bwq_features.parquet
```

**MFE only (requires parquet with sequences):**

```bash
python feature_wrapper.py mfe \
    --input sequences.parquet \
    --sequence-col Sequence \
    --include-structure \
    --output mfe_features.parquet
```

**K-mer only:**

```bash
python feature_wrapper.py kmer \
    --input sequences.fasta \
    --k-min 3 --k-max 12 \
    --output-format sparse_dataframe \
    --output kmer_features.parquet
```

#### 4\. Using Saved K-mer Artifacts

Reuse previously computed k-mer sparse matrix:

```bash
python feature_wrapper.py all \
    --gtf annotations.gtf \
    --ref-genome genome.fa \
    --bwq-config tracks.yaml \
    --use-saved-kmer-base /path/to/kmer_artifacts \
    --use-dim-redux --redux-n-components 1 \
    --output features.parquet
```

#### 5\. Cache Management

**View cache information:**

```bash
python feature_wrapper.py cache info
```

**Clear cache:**

```bash
python feature_wrapper.py cache clear
```

### 1b. Command Reference

#### Global Options

  - `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - `--threads`, `-t`: Number of worker threads/processes
  - `--cache-dir`: Directory for caching files (default: `./bwq_tracks`)
  - `--no-cache`: Disable file caching
  - `--clear-cache`: Clear cache before running
  - `--show-progress` / `--no-progress`: Control progress bar display
  - `--quiet`: Suppress progress bars and most logs

#### Command: `all` (Full Pipeline)

**Required Arguments:**

  - `--gtf`: GTF annotation file (for sequence extraction and BED generation)
  - `--ref-genome`: Reference genome FASTA file
  - `--bwq-config`: BigWig/BigBed configuration YAML file

**Optional Arguments:**

  - `--bed`: Pre-existing BED file (auto-generated from GTF if omitted)
  - `--fasta`: Pre-extracted FASTA sequences (auto-extracted from GTF if omitted)
  - `--output`, `-o`: Output parquet file path
  - `--k-min`: Minimum k-mer length (default: 3)
  - `--k-max`: Maximum k-mer length (default: 12)
  - `--use-dim-redux`: Enable SVD dimensionality reduction
  - `--redux-n-components`: Number of SVD components (default: 1)
  - `--use-tfidf`: Apply TF-IDF transformation
  - `--sparse`: Keep k-mer features as sparse matrix
  - `--no-group-kmer-by-length`: Use global SVD instead of grouped per k-length
  - `--kmer-sparse-base`: Base path to save raw k-mer artifacts
  - `--use-saved-kmer-base`: Base path to load pre-computed k-mer artifacts
  - `--return-kmer-sparse-paths`: Return k-mer artifact paths in output

#### Command: `bwq` (BigWig Query)

  - `--bed`: BED file with genomic regions (required)
  - `--bwq-config`: BigWig/BigBed configuration file (required)
  - `--output`, `-o`: Output parquet file

#### Command: `mfe` (RNA Secondary Structure)

  - `--input`: Input parquet file with sequences (required)
  - `--sequence-col`: Column name containing RNA sequences (default: "Sequence")
  - `--include-structure`: Include predicted structure in output
  - `--num-processes`: Number of parallel processes
  - `--output`, `-o`: Output parquet file

#### Command: `kmer` (K-mer Profiling)

  - `--input`: FASTA file or directory (required)
  - `--k-min`: Minimum k-mer length (default: 3)
  - `--k-max`: Maximum k-mer length (default: 12)
  - `--output-format`: Output format (dataframe, sparse\_dataframe, matrix)
  - `--output`, `-o`: Output parquet file
  - `--return-sparse-paths`: Persist raw sparse artifacts
  - `--sparse-base`: Base path for sparse artifacts

#### Command: `cache` (Cache Management)

  - `cache info`: Show cache information
  - `cache clear`: Clear the cache

### 1c. Feature Details & Advanced Features

#### 🧬 GTF-Based Workflow

The pipeline is designed around **GTF annotation files** for maximum accuracy:

  - **Accurate Splicing**: Properly splices multi-exon transcripts via gffutils (concatenating exons).
  - **Strand-Aware**: Automatically handles strand orientation (reverse complement for minus strand).
  - **DNA-to-RNA**: Converts DNA to RNA (T → U).
  - **Auto-Generation**: Automatically generates BED files from the GTF for BWQ processing.
  - **Caching**: Caches gffutils databases for faster reruns on the same GTF file.

#### 📁 Cache & Performance

  - **Efficient Caching**: Persistently caches remote BigWig/BigBed files (configurable via `--cache-dir`) and gffutils databases.
  - **Parallel Processing**: Utilizes parallel processing across all modules (BWQ, MFE, K-mer).
  - **Memory-Efficient**: Employs sparse matrix support for k-mer features to handle high dimensionality.
  - **Artifact Reuse**: Allows saving (`--kmer-sparse-base`) and reusing (`--use-saved-kmer-base`) k-mer artifacts to skip re-computation.

#### ⚙️ K-mer Transformations

  - **TF-IDF Weighting** (`--use-tfidf`): Modulates raw k-mer frequency counts to give more weight to informative k-mers.
  - **Dimensionality Reduction** (`--use-dim-redux`): Uses Truncated SVD to reduce the high-dimensional k-mer space.
  - **Grouped SVD**: By default, SVD is performed *per k-length* for a more balanced representation. Use `--no-group-kmer-by-length` for a single global decomposition.
  - **Sparse Storage**: Transformed matrices can be stored in the DataFrame `attrs` (`--sparse`) to minimize memory and keep the main DataFrame clean.

#### 🔗 Feature Integration

  - **Automatic Merging**: All extracted features (BWQ, MFE, K-mer) are automatically merged on transcript IDs.
  - **Smart Renaming**: Columns are renamed to avoid conflicts.
  - **Length Calculation**: Transcript lengths are calculated from genomic coordinates.

### 1d. BigWig/BigBed Configuration

The BWQ configuration file (`--bwq-config`) is a YAML file that specifies signal tracks and statistics to extract. See `example_config.yaml` for a template.

**Configuration Structure:**

```yaml
# List of BigWig/BigBed files to query
- path: /path/to/file.bigWig      # Local path or URL
  upstream: 1000                  # Optional: extend upstream
  downstream: 1000                # Optional: extend downstream
  stats:                          # Statistics to calculate
    - stat: mean                  # Statistic type
      name: h3k27ac_mean          # Output column name
    - stat: max
      name: h3k27ac_max
    - stat: coverage
      name: h3k27ac_coverage

- path: [https://example.com/file.bigBed](https://example.com/file.bigBed)
  stats:
    - stat: coverage
      name: peaks_coverage
```

**Available Statistics:**

  - `mean`, `max`, `min`, `sum`: Numerical summaries
  - `coverage`: Fraction of region covered
  - `extract_names`: Extract names from BigBed entries (with `name_field_index`)

### 1e. Python API (Extraction)

#### Basic Usage

```python
from features.feature_wrapper import FeatureWrapper

# Initialize wrapper
fw = FeatureWrapper(
    cache_dir="./bwq_tracks",
    keep_downloaded_files=True,
    threads=8
)

# Run complete pipeline (GTF-based)
df = fw.run_all(
    gtf_file="annotations.gtf",
    ref_genome_path="genome.fa",
    config_file="bwq_config.yaml",
    k_min=3, k_max=12,
    use_dim_redux=True,
    redux_n_components=1,
    use_tfidf=True,
    sparse=True
)
```

#### Individual Module Execution

```python
# BigWig Query
bwq_df = fw.run_bwq(
    bed_file="regions.bed",
    config_file="bwq_config.yaml"
)

# MFE (requires DataFrame with 'Sequence' column)
mfe_df = fw.run_mfe(
    df_input=sequence_df,
    sequence_col="Sequence",
    include_structure=True
)

# K-mer profiling
kmer_df = fw.run_kmer(
    input_path="sequences.fasta",
    k_min=3, k_max=12,
    output_format="sparse_dataframe"
)
```

#### GTF-Based Sequence Extraction

```python
# Extract spliced transcript sequences from GTF
fasta_path, seq_df = fw.extract_transcripts_from_gtf(
    gtf_file="annotations.gtf",
    ref_genome_path="genome.fa",
    output_fasta="transcripts.fasta",
    return_df=True
)

# Auto-generate BED from GTF
bed_path = fw.gtf_transcripts_to_bed(
    gtf_file="annotations.gtf",
    output_bed="transcripts.bed"
)
```

#### Cache Management

```python
# Get cache information
cache_info = fw.get_cache_info()
print(f"Cache size: {cache_info['cache_size_mb']} MB")
print(f"Cached files: {cache_info['cached_files_count']}")

# Clear cache
fw.clear_cache()
```

-----

## Part 2: Feature Cleaning (`feature_cleaning.py`)

Data preparation pipeline for model training and inference.

### 2a. Usage Examples

#### Training Mode

Prepare data for model training with automatic train/val/test splits:

```bash
python feature_cleaning.py \
    --mode training \
    --dataset features.parquet \
    --label-column is_lncrna \
    --output-dir prepared_data/ \
    --split-suffix "v1" \
    --scaler standard \
    --train-size 0.7 --val-size 0.15 --test-size 0.15 \
    --enable-multi-hot \
    --scaler-path prepared_data/scaler.pkl
```

**Outputs:**

  - `prepared_data/cleaned_dataset_v1.parquet`
  - `prepared_data/X_train_v1.parquet`, `X_val_v1.parquet`, `X_test_v1.parquet`
  - `prepared_data/scaler.pkl`

#### Inference Mode

Prepare new data for predictions using a saved model schema:

```bash
python feature_cleaning.py \
    --mode inference \
    --dataset new_transcripts.parquet \
    --output-dir inference_data/ \
    --split-suffix "inference" \
    --metadata-path model/flync_model_schema.json \
    --scaler-path prepared_data/scaler.pkl \
    --scaler standard
```

**Outputs:**

  - `inference_data/cleaned_dataset_inference.parquet`
  - `inference_data/schema_diff_inference.txt`
  - `inference_data/schema_diff_inference.json`

#### Validation Mode

Dry-run to validate data without saving outputs:

```bash
python feature_cleaning.py \
    --mode validation \
    --dataset features.parquet \
    --label-column is_lncrna \
    --metadata-path model/flync_model_schema.json
```

#### Paired Input Mode

Provide separate positive and negative class datasets:

```bash
python feature_cleaning.py \
    --mode training \
    --positive-path lncrna_features.parquet \
    --negative-path protein_coding_features.parquet \
    --label-column is_lncrna \
    --output-dir prepared_data/ \
    --split-suffix "paired_v1" \
    --scaler minmax
```

### 2b. Command Reference

**Mode Selection (Required):**

  - `--mode {training,inference,validation}`: Set the operational mode.

**Input/Output:**

  - `--dataset`: Path to the input feature parquet file.
  - `--positive-path`, `--negative-path`: Paths for paired input mode.
  - `--output-dir`: Directory to save cleaned data and splits.
  - `--split-suffix`: Suffix to append to output files (e.g., "v1").
  - `--label-column`: Name of the target label column.

**Data Cleaning:**

  - `--enable-multi-hot`: Apply multi-hot encoding to categorical features.
  - `--prefix-multi-hot`: Prefix encoded columns with original column name.

**Scaling:**

  - `--scaler {standard,minmax,none}`: Feature scaling method.
  - `--scaler-path`: Path to save (training) or load (inference) the fitted scaler.

**Splits (training mode only):**

  - `--train-size`: Proportion for training (default: 0.7).
  - `--val-size`: Proportion for validation (default: 0.15).
  - `--test-size`: Proportion for testing (default: 0.15).
  - `--random-seed`: Random seed for reproducible splits (default: 42).

**Schema Validation (inference/validation mode):**

  - `--metadata-path`: Path to the model schema JSON for validation.

### 2c. Data Cleaning Details

**Missing Value Strategies:**

1.  **Signal features** (min\_, max\_, mean\_, std\_, sum\_, cov\_, cpat\_): Fill with 0
2.  **Structure features** (ss\_):
      * `ss_mfe`: Fill with 0
      * Other `ss_` features: Drop column if \>50% missing, else fill with median
3.  **K-mer features** (starts with digit or contains "mer\_SVD"): Fill with 0
4.  **Categorical features**: Fill with "unknown"

**Columns Dropped:**

  - All CPAT features (`cpat_*`)
  - Extra secondary structure features (all `ss_*` except `ss_mfe`)
  - Genomic coordinates (`chromosome`, `start`, `end`)

**Column Sanitization:**

  - Remove special characters and spaces
  - Convert to lowercase
  - Replace consecutive underscores with single underscore
  - Strip leading/trailing underscores

### 2d. Python API (Cleaning)

```python
from features.feature_cleaning import prepare_training_data, PipelineConfig
from pathlib import Path

# Training mode
config = PipelineConfig(
    mode="training",
    dataset_path=Path("features.parquet"),
    label_column="is_lncrna",
    split_output_dir=Path("prepared_data"),
    split_suffix="v1",
    scaler="standard",
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    enable_multi_hot=True,
    scaler_path=Path("prepared_data/scaler.pkl")
)

result = prepare_training_data(config)

# Access results
print(f"Cleaned dataset: {result.cleaned_path}")
print(f"Train split: {result.split_paths['train']}")
print(f"Scaler saved: {result.scaler_path}")

# Inference mode
config_inference = PipelineConfig(
    mode="inference",
    dataset_path=Path("new_data.parquet"),
    split_output_dir=Path("inference_data"),
    split_suffix="inference",
    metadata_path=Path("model/flync_model_schema.json"),
    scaler="standard",
    scaler_path=Path("prepared_data/scaler.pkl")
)

result = prepare_training_data(config_inference)

# Check schema alignment
if result.schema_diff:
    print(f"Missing columns: {result.schema_diff.missing_columns}")
    print(f"Extra columns: {result.schema_diff.extra_columns}")
    print(f"Type mismatches: {result.schema_diff.dtype_mismatches}")
```

-----

## General Information

### Best Practices

1.  **Always use GTF files** as primary input for accurate spliced sequence extraction.
2.  **Enable caching** (`--cache-dir`) to avoid re-downloading BigWig files.
3.  **Use sparse format** (`--sparse`) for large k-mer ranges to save memory.
4.  **Tune thread counts** (`--threads`) based on available CPU cores.
5.  **Save k-mer artifacts** (`--kmer-sparse-base`) for reuse in multiple experiments.
6.  **Enable transformations** (`--use-tfidf --use-dim-redux`) to reduce k-mer dimensionality and improve signal.
7.  **Use Grouped SVD** (default) for a better k-mer representation.

### Troubleshooting

#### Dependencies

Required program:
  - https://github.com/LinearFold/LinearFold

Required Python packages:

  - `pandas`
  - `pyranges`
  - `scipy`
  - `gffutils`
  - `pyfaidx`

Feature modules require their specific dependencies (e.g., `bwq` for BigWig, `mfe` for structure, `kmer` for counting).

#### Common Issues

**"GTF file missing 'Feature' column"**

  - Ensure your GTF file follows standard format with feature type in column 3.

**"No sequences available for downstream feature generation"**

  - Check that your GTF file contains transcript and exon features.
  - Verify reference genome FASTA is accessible and indexed.

**"kmer\_redux utilities not available"**

  - Ensure `utils/kmer_redux.py` module is available in your installation.
  - K-mer transformations (TF-IDF/SVD) require this module.

**"No transcripts found in GTF file"**

  - Check that your GTF has entries with feature\_type="transcript".
  - Consider filtering your GTF to retain only transcript-level features.
