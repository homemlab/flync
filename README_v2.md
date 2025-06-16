# FLYNC v2.0 - Refactored Pipeline

This is the refactored version of FLYNC using modern Python practices and industry-standard bioinformatics pipeline management.

## Key Improvements

- **Python-first approach**: All logic implemented in Python with proper modularity
- **Modern CLI**: Using Typer for beautiful command-line interface
- **Configuration management**: YAML-based configuration with validation
- **Workflow orchestration**: Structured pipeline with resume capability
- **Machine learning module**: Separated ML components for retraining and prediction
- **Better error handling**: Comprehensive logging and error recovery
- **Industry standards**: Following bioinformatics best practices

## Installation

### Prerequisites

- Python 3.8+
- Bioinformatics tools (HISAT2, StringTie, etc.)
- Conda/Mamba for environment management

### Install from source

```bash
# Clone the repository
git clone https://github.com/homemlab/flync.git
cd flync

# Install in development mode
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

## Usage

### Configuration-based execution (Recommended)

1. Create a configuration file:
```bash
flync init-config --output my_config.yaml
```

2. Edit the configuration file to match your needs

3. Run the pipeline:
```bash
flync run --config my_config.yaml
```

### Quick execution modes

#### SRA mode
```bash
flync sra --list sra_accessions.txt --output results/ --metadata metadata.csv --threads 4
```

#### FASTQ mode
```bash
flync fastq --fastq /path/to/fastq/ --output results/ --paired --threads 4
```

## Configuration

Example configuration file:

```yaml
# Basic settings
threads: 4
memory_gb: 16

# Input data
input:
  sra_list: "data/sra_list.txt"  # OR fastq_dir for FASTQ mode
  # fastq_dir: "/path/to/fastq"
  # paired_end: true
  metadata: "data/metadata.csv"  # Optional, for differential expression

# Output settings
output:
  output_dir: "./results"
  keep_intermediate_files: false
  log_level: "INFO"

# Genome settings
genome:
  species: "drosophila"
  release: "BDGP6.32"
  download_if_missing: true
  # custom_genome: "/path/to/genome.fa"  # Optional
  # custom_annotation: "/path/to/annotation.gtf"  # Optional

# Tool parameters
tools:
  hisat2_threads: 2
  hisat2_extra_args: ""
  stringtie_threads: 2
  stringtie_extra_args: ""
  prediction_threshold: 0.5

# Advanced options
resume: false
dry_run: false
```

## Pipeline Steps

1. **Genome Preparation**: Download and prepare reference files
2. **Input Preparation**: Process SRA accessions or validate FASTQ files
3. **Read Mapping**: HISAT2 alignment to reference genome
4. **Transcriptome Assembly**: StringTie assembly and merging
5. **Coding Probability**: Calculate coding potential and classify transcripts
6. **Differential Expression**: Optional DE analysis with metadata
7. **Feature Extraction**: Extract genomic and sequence features
8. **ML Prediction**: Random Forest classification of lncRNAs

## Machine Learning Module

### Retraining the model

```python
from flync.ml import retrain_model

metrics = retrain_model(
    positive_samples_path="data/lncrna_features.csv",
    negative_samples_path="data/coding_features.csv", 
    output_model_path="models/new_model.pkl",
    n_estimators=200
)
```

### Using a custom model

Set the model path in your configuration:

```yaml
tools:
  ml_model_path: "/path/to/custom_model.pkl"
```

## Development

### Project Structure

```
src/flync/
├── __init__.py           # Package initialization
├── cli.py               # Modern CLI interface
├── config.py            # Configuration management
├── pipeline/            # Base pipeline framework
├── workflows/           # Workflow orchestration
│   ├── main_workflow.py # Main workflow logic
│   └── steps.py         # Individual pipeline steps
├── ml/                  # Machine learning components
│   └── __init__.py      # ML classifier and utilities
└── utils/               # Utility functions
    ├── __init__.py      # Logging utilities
    └── file_utils.py    # File operations
```

### Running tests

```bash
pytest tests/
```

### Code formatting

```bash
black src/
isort src/
```

## Migration from v1.x

The refactored version maintains compatibility with the original configuration format. Key changes:

- Replace `./flync sra` with `flync sra`
- Use YAML configuration files instead of command-line arguments
- New CLI commands: `flync init-config`, `flync run`
- Enhanced logging and error reporting

## Advanced Features

- **Resume capability**: Automatically resume from failed steps
- **Dry-run mode**: Preview pipeline execution without running
- **Configurable parameters**: All tool parameters are configurable
- **Custom genomes**: Support for user-provided reference files
- **Model retraining**: Built-in ML model retraining capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
