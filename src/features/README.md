# Enhanced Feature Orchestrator

The Enhanced Feature Orchestrator provides flexible, feature-specific input handling and organized output management for comprehensive genomic feature extraction.

## Key Features

### 🔧 **Feature-Specific Inputs**
- `--bwq-input`: BED file for BigWig/BigBed statistics
- `--mfe-input`: FASTA/CSV/Parquet for RNA secondary structure MFE
- `--cpat-input`: FASTA for coding potential assessment  
- `--kmer-input`: FASTA file or directory for k-mer profiles

### 📁 **Organized Output Structure**
```
output_directory/
├── bwq/
│   └── bwq_features.parquet
├── mfe/
│   └── mfe_features.parquet
├── cpat/
│   └── cpat_features.parquet
├── kmer/
│   ├── kmer_features_binary_sparse.npz
│   ├── kmer_features_binary_rows.txt
│   └── kmer_features_binary_cols.txt
└── unified_features.parquet
```

### ⚙️ **Configuration File Support**
Use YAML or JSON configuration files to define all parameters:

```yaml
# feature_extraction_config.yaml
bwq_config: /path/to/bwq_tracks.yaml
hexamer_table: /path/to/hexamer_table.tsv
logit_model: /path/to/logit_model.pkl
reference_fasta: /path/to/reference.fa

features:
  bwq:
    threads: 4
  mfe:
    workers: 4
    batch_size: 1000
  cpat:
    workers: 4
  kmer:
    k_min: 3
    k_max: 12
    output_format: sparse
    workers: 4
    batch_size: 10000

unify: true
keep_intermediates: false
```

## Usage Examples

### 1. **Feature-Specific Extraction**
Extract different features from different input files:

```bash
python orchestrator_v2.py --output-dir results/ \
    --bwq-input genomic_ranges.bed --bwq-config tracks.yaml \
    --mfe-input sequences.fasta \
    --cpat-input sequences.fasta --cpat-hexamer hex.tsv --cpat-model model.pkl \
    --kmer-input sequences.fasta
```

### 2. **Using Existing Features**
Combine existing k-mer sparse matrix with new MFE extraction:

```bash
python orchestrator_v2.py --output-dir results/ \
    --kmer-input /path/to/existing/sparse_matrix/ \
    --mfe-input new_sequences.parquet
```

### 3. **Configuration File Mode**
Use a configuration file for complex setups:

```bash
# Create sample config
python orchestrator_v2.py --create-config

# Edit the config file, then run
python orchestrator_v2.py --config feature_extraction_config.yaml --output-dir results/
```

### 4. **Single Feature Extraction**
Extract only specific features with custom parameters:

```bash
# K-mer only with custom range
python orchestrator_v2.py --output-dir results/ \
    --kmer-input sequences.fasta \
    --kmer-k-min 4 --kmer-k-max 8 \
    --kmer-format dense

# MFE only with performance tuning
python orchestrator_v2.py --output-dir results/ \
    --mfe-input sequences.fasta \
    --mfe-workers 8 --mfe-batch-size 2000
```

### 5. **CPAT with Reference**
Extract CPAT features with all required parameters:

```bash
python orchestrator_v2.py --output-dir results/ \
    --cpat-input sequences.fasta \
    --cpat-hexamer hexamer_table.tsv \
    --cpat-model logistic_model.pkl \
    --cpat-workers 8
```

## Parameter Reference

### Global Parameters
- `--output-dir`: Output directory for organized results
- `--config`: Configuration file (YAML/JSON)
- `--no-unify`: Skip automatic feature unification
- `--keep-intermediates`: Keep temporary files
- `--log-level`: Logging verbosity

### BWQ Parameters
- `--bwq-input`: BED file input
- `--bwq-config`: BWQ configuration file
- `--bwq-threads`: Number of processing threads

### MFE Parameters  
- `--mfe-input`: Input file (FASTA/CSV/Parquet)
- `--mfe-workers`: Number of worker processes
- `--mfe-batch-size`: Processing batch size

### CPAT Parameters
- `--cpat-input`: FASTA input file
- `--cpat-hexamer`: Hexamer frequency table
- `--cpat-model`: Logistic regression model
- `--cpat-ref`: Reference FASTA (alias for hexamer)
- `--cpat-workers`: Number of worker processes

### K-mer Parameters
- `--kmer-input`: FASTA input or existing directory
- `--kmer-k-min`: Minimum k-mer length (default: 3)
- `--kmer-k-max`: Maximum k-mer length (default: 12)
- `--kmer-format`: Output format (sparse/dense)
- `--kmer-workers`: Number of worker processes
- `--kmer-batch-size`: Processing batch size

## Advanced Features

### 🔄 **Automatic Input Conversion**
- FASTA → CSV conversion for MFE extraction
- Format validation and preprocessing
- Intelligent file type detection

### 🚀 **Performance Optimization**
- Parallel processing for all feature types
- Configurable batch sizes
- Memory-efficient sparse matrix handling
- Optional dimensionality reduction

### 🔗 **Smart Feature Unification**
- Automatic dataset merging on common identifiers
- Sparse matrix integration
- Conflict resolution for overlapping columns
- Comprehensive error handling

### 📊 **Output Flexibility**
- Organized directory structure
- Individual feature outputs
- Unified consolidated dataset
- Metadata preservation

## Migration from Previous Version

The enhanced orchestrator provides a comprehensive interface:

```bash
# Current interface
python orchestrator.py --output-dir results/ --mfe-input input.fasta

# Configuration-driven approach
python orchestrator.py --config config.yaml --output-dir results/
```

## Best Practices

1. **Use configuration files** for complex workflows
2. **Organize inputs** by feature type for clarity
3. **Monitor resource usage** with worker/thread parameters
4. **Keep intermediates** during development/debugging
5. **Validate outputs** in the organized directory structure

## Performance Tips

- Use `--kmer-format sparse` for large k-mer ranges
- Adjust `--*-workers` based on available CPU cores
- Use `--*-batch-size` to control memory usage
- Consider `--no-unify` for very large datasets

---

The Enhanced Feature Orchestrator v2 provides a powerful, flexible platform for comprehensive genomic feature extraction with enterprise-grade organization and performance capabilities.
