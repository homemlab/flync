![FLYNC logo](logo.jpeg)

# FLYNC - FLY Non-Coding gene discovery & classification

FLYNC is an end-to-end bioinformatics pipeline for discovering and classifying non-coding genes in *Drosophila melanogaster*. It combines RNA-seq processing, feature extraction from genomic databases, and machine learning prediction to identify and classify lncRNA candidates.

- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Install from Source](#install-from-source)
  - [Docker](#docker)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Setup Reference Genome](#1-setup-reference-genome)
  - [Configure Pipeline](#2-configure-pipeline)
  - [Run Bioinformatics Pipeline](#3-run-bioinformatics-pipeline)
  - [Run ML Prediction](#4-run-ml-prediction)
- [Pipeline Details](#pipeline-details)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## Pipeline Overview

FLYNC performs the following steps:

1. **Read Mapping** - Align RNA-seq reads to reference genome (HISAT2)
2. **Transcriptome Assembly** - Reconstruct transcripts per sample (StringTie)
3. **Assembly Merging** - Create unified transcriptome across samples
4. **Feature Extraction** - Query genomic tracks, k-mers, and structural features
5. **ML Classification** - Predict lncRNA candidates using trained Random Forest model

The pipeline is now built on modern Python tools:
- **Snakemake** for workflow management
- **Click** for CLI interface
- **Scikit-learn** for machine learning
- **Pandas/PyBigWig** for genomic data processing

## Installation

### Prerequisites

- Linux-based operating system (tested on Debian/Ubuntu)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/)
- Git
- 8+ GB RAM recommended
- 20+ GB disk space for genome and indices

### Install from Source

1. **Clone the repository:**
   ```bash
   git clone https://github.com/homemlab/flync.git
   cd flync
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate flync
   ```

3. **Install Python package:**
   ```bash
   pip install -e .
   ```

4. **Verify installation:**
   ```bash
   flync --help
   ```

### Docker

Pre-built Docker images are available for easy deployment:

```bash
# Standard image
docker pull rfcdsantos/flync:latest

# With embedded genomic tracks (faster, larger)
docker pull rfcdsantos/flync:local-tracks
```

**Usage with Docker:**
```bash
docker run --rm -v $PWD:/data rfcdsantos/flync \
  flync run-bio --configfile /data/config/config.yaml
```

## Quick Start

```bash
# 1. Setup reference genome and indices
flync setup --genome-dir genome

# 2. Generate configuration template
flync config --template --output my_config.yaml

# 3. Edit my_config.yaml with your sample information

# 4. Run bioinformatics pipeline
flync run-bio --configfile my_config.yaml --cores 8

# 5. Predict lncRNAs from assembled transcripts
flync run-ml \
  --gtf results/assemblies/merged-new-transcripts.gtf \
  --output results/lncrna_predictions.csv \
  --ref-genome genome/genome.fa
```

## Usage

### 1. Setup Reference Genome

Download *D. melanogaster* BDGP6.32 (dm6) genome and build HISAT2 indices:

```bash
flync setup --genome-dir genome --build-index
```

This downloads:
- Reference genome (BDGP6.32/dm6) from Ensembl
- Gene annotation (GTF) release 106
- Builds HISAT2 indices
- Extracts splice sites

**Skip download if files exist:**
```bash
flync setup --skip-download
```

### 2. Configure Pipeline

Generate a configuration template:

```bash
flync config --template --output config/my_analysis.yaml
```

Edit `my_analysis.yaml` to specify:
- **samples**: Path to sample list file (OPTIONAL - see below)
- **genome**: Path to reference FASTA
- **annotation**: Path to reference GTF
- **output_dir**: Where to save results
- **threads**: Number of CPU cores to use

**Sample Specification (3 options):**

**Option 1: Auto-detect from FASTQ directory (simplest!)**
```yaml
samples: null  # Set to null to enable auto-detection
fastq_dir: "/path/to/fastq/files"
fastq_paired: false  # true for paired-end, false for single-end
```
Samples are automatically detected from FASTQ filenames!

**Option 2: Plain text list**
```yaml
samples: "samples.txt"
```
```txt
# samples.txt
sample1
sample2
sample3
```

**Option 3: CSV with metadata (for DGE analysis)**
```yaml
samples: "samples.csv"
```
```csv
sample_id,condition,replicate
sample1,control,1
sample2,control,2
sample3,treated,1
```

Use Option 1 for quick processing, Option 2 to control which samples run, Option 3 for differential expression analysis.
sample_id,condition,replicate
SRR1234567,control,1
SRR1234568,control,2
SRR1234569,treated,1
SRR1234570,treated,2
```

### 3. Run Bioinformatics Pipeline

Execute the complete RNA-seq analysis workflow:

**Option A: Using SRA accessions (default)**

```bash
flync run-bio --configfile config/my_analysis.yaml --cores 8
```

Your `samples.csv` should contain SRA accession numbers:
```csv
sample_id,condition,replicate
SRR1234567,control,1
SRR1234568,control,2
SRR1234569,treated,1
SRR1234570,treated,2
```

**Option B: Using local FASTQ files**

If you already have FASTQ files downloaded, configure them in your `config.yaml`:

```yaml
# config.yaml
samples: null  # Set to null for auto-detection
fastq_dir: "/path/to/fastq/files"
fastq_paired: false  # true for paired-end, false for single-end
```

Then run:
```bash
flync run-bio --configfile config.yaml --cores 8
```

The pipeline will automatically detect all samples from your FASTQ filenames. No sample list file required!

**File naming for auto-detection:**
- **Paired-end:** `sample1_1.fastq.gz`, `sample1_2.fastq.gz` → detects `sample1`
- **Single-end:** `sample1.fastq.gz` → detects `sample1`

**Alternative: Specify samples explicitly**

If you want to process only specific samples or need metadata for differential expression, create a sample list:

```bash
# Option 1: Plain text list
cat > samples.txt << EOF
sample1
sample2
EOF

# Option 2: CSV with metadata for DGE
cat > samples.csv << EOF
sample_id,condition,replicate
sample1,control,1
sample2,treated,1
EOF
```

Then update `config.yaml`:
```yaml
samples: "samples.txt"  # or "samples.csv"
fastq_dir: "/path/to/fastq/files"
fastq_paired: false
```

**Options:**
- `--dry-run`: Show what would be executed without running
- `--unlock`: Unlock working directory after a crash
- `--cores`: Number of CPU cores (default: 8)

**Configuration in `config.yaml`:**
- Set `samples: null` for auto-detection, or provide a CSV/TXT file path
- Set `fastq_dir` and `fastq_paired` for local FASTQ files
- Leave `fastq_dir` unset to download from SRA (requires samples CSV with SRA accessions)

**Output files:**
- `results/data/{sample}/{sample}.sorted.bam` - Aligned reads
- `results/assemblies/stringtie/{sample}.rna.gtf` - Per-sample assemblies
- `results/assemblies/merged.gtf` - Unified transcriptome
- `results/assemblies/merged-new-transcripts.gtf` - Novel transcripts only
- `results/cuffcompare/cuffcomp.gtf.stats` - Assembly comparison stats

### 4. Run ML Prediction

Classify lncRNA candidates from assembled transcripts:

```bash
flync run-ml \
  --gtf results/assemblies/merged-new-transcripts.gtf \
  --output results/lncrna_predictions.csv \
  --ref-genome genome/genome.fa \
  --threads 8
```

**Options:**
- `--gtf`: Input GTF file (merged.gtf or merged-new-transcripts.gtf)
- `--output`: Output CSV file for predictions
- `--model`: Path to custom model (default: uses bundled model)
- `--ref-genome`: Reference genome FASTA
- `--bwq-config`: BigWig query configuration (optional)
- `--threads`: Number of threads for feature extraction

**Output format:**
```csv
transcript_id,prediction,confidence,probability_lncrna
MSTRG.1.1,1,0.95,0.95
MSTRG.1.2,0,0.87,0.13
```

Where:
- `prediction`: 1 = lncRNA, 0 = protein-coding
- `confidence`: Model confidence score
- `probability_lncrna`: Probability of being lncRNA

## Pipeline Details

### Feature Extraction

The ML model uses the following features:

**Sequence Features:**
- K-mer frequencies (3-12 mers) with TF-IDF transformation
- GC content
- Transcript length
- Exon count

**Genomic Track Features:**
- CAGE-seq (TSS evidence)
- H3K4me3 ChIP-seq (promoter marks)
- Pol2 ChIP-seq (transcription)
- Conservation scores (phyloP, phastCons)
- TF binding sites (JASPAR, ReMap)

**Structural Features:**
- RNA folding energy (RNAfold MFE)

### Machine Learning Model

- **Algorithm**: Random Forest classifier
- **Training Data**: 
  - Positive: Known *D. melanogaster* lncRNAs from NONCODE, lncRNAdb
  - Negative: Protein-coding genes from FlyBase
- **Performance** (on holdout set):
  - Precision: ~0.90
  - Recall: ~0.85
  - F1-score: ~0.87

## Advanced Usage

### Custom Configuration

**Using local FASTQ files instead of SRA:**
```yaml
# In config.yaml
samples: "local_samples.csv"
fastq_dir: "/path/to/fastq/files"
```

**Adjust tool parameters:**
```yaml
params:
  hisat2: "-p 16 --dta --very-sensitive"
  stringtie_assemble: "-p 16 -f 0.1"
  stringtie_merge: "-m 200 -f 0.01"
```

### Feature Extraction Only

If you already have a GTF and want features only:

```bash
python src/flync/features/feature_wrapper.py all \
  --gtf merged.gtf \
  --ref-genome genome/genome.fa \
  --bwq-config config/bwq_tracks.yaml \
  --use-tfidf --use-dim-redux \
  --output features.parquet
```

### Prediction from Pre-Extracted Features

```bash
python -c "
from flync.ml.predictor import predict_from_features
predict_from_features(
    'features.parquet',
    'src/flync/assets/rf_dm6_lncrna_classifier.model',
    'predictions.csv'
)
"
```

### Hyperparameter Optimization

Train a new model with custom parameters:

```bash
# Prepare data splits
python src/flync/optimizer/prepare_data.py \
  --positive-file lncrna_features.parquet \
  --negative-file protein_coding_features.parquet \
  --output-dir datasets/

# Optimize hyperparameters
python src/flync/optimizer/hyperparameter_optimizer.py \
  --train-data datasets/train.parquet \
  --test-data datasets/test.parquet \
  --model-type randomforest \
  --n-trials 100 \
  --experiment-name "My_RF_Model"

# View results
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

**Problem**: `command not found: flync`
- **Solution**: Activate conda environment: `conda activate flync`

**Problem**: `Snakefile not found`
- **Solution**: Reinstall package: `pip install -e .`

**Problem**: HISAT2 index build fails
- **Solution**: Check available disk space (requires ~10GB) and memory (4GB+)

**Problem**: Feature extraction fails with "track not accessible"
- **Solution**: Check internet connection. Genomic tracks are downloaded from UCSC/Ensembl

**Problem**: ML prediction error "schema mismatch"
- **Solution**: Ensure feature extraction uses same transformations as training:
  ```bash
  --use-tfidf --use-dim-redux --redux-n-components 1
  ```

**Problem**: Docker permission denied
- **Solution**: Add user to docker group or run with `sudo`

## File Structure

```
flync/
├── src/flync/              # Main Python package
│   ├── cli.py             # Command-line interface
│   ├── workflows/         # Snakemake workflow
│   │   ├── Snakefile
│   │   └── rules/         # Workflow rules
│   ├── features/          # Feature extraction
│   ├── ml/                # ML prediction
│   ├── optimizer/         # Hyperparameter tuning
│   ├── utils/             # Utilities
│   └── assets/            # Bundled models
├── config/                # Configuration files
├── genome/                # Reference genome files
├── test/                  # Test data
├── environment.yml        # Conda environment
├── pyproject.toml         # Python package metadata
└── README.md
```

## Migrating from v1

If you're upgrading from the bash-based version:

1. **Backup old files:**
   ```bash
   ./deprecate.sh
   ```

2. **Reinstall environment:**
   ```bash
   conda env remove -n flync
   conda env create -f environment.yml
   conda activate flync
   pip install -e .
   ```

3. **Update workflows:**
   - Replace `./flync sra ...` with `flync run-bio ...`
   - Replace direct script calls with `flync run-ml ...`
   - Use YAML config files instead of command-line arguments

## Citation

If you use FLYNC in your research, please cite:

```bibtex
@software{flync2024,
  title={FLYNC: lncRNA discovery pipeline for Drosophila melanogaster},
  author={Santos, R. and contributors},
  year={2024},
  url={https://github.com/homemlab/flync}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- **Issues**: https://github.com/homemlab/flync/issues
- **Documentation**: https://github.com/homemlab/flync/wiki
- **Contact**: Open an issue for questions

---

**Version**: 1.0.0 (Python-first refactoring)  
**Last Updated**: November 2025
