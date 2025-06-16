"""Configuration management for FLYNC pipeline."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator


class GenomeConfig(BaseModel):
    """Configuration for genome reference files."""
    
    species: str = "drosophila"
    release: str = "BDGP6.32"
    download_if_missing: bool = True
    custom_genome: Optional[Path] = None
    custom_annotation: Optional[Path] = None


class ToolConfig(BaseModel):
    """Configuration for bioinformatics tools."""
    
    # HISAT2 parameters
    hisat2_threads: int = Field(default=1, ge=1)
    hisat2_extra_args: str = ""
    
    # StringTie parameters
    stringtie_threads: int = Field(default=1, ge=1)
    stringtie_extra_args: str = ""
    
    # Feature extraction parameters
    feature_extraction_threads: int = Field(default=1, ge=1)
    
    # Machine learning parameters
    ml_model_path: Optional[Path] = None
    prediction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class InputConfig(BaseModel):
    """Configuration for input data."""
    
    # SRA mode
    sra_list: Optional[Path] = None
    
    # FASTQ mode
    fastq_dir: Optional[Path] = None
    paired_end: bool = False
    
    # Metadata for differential expression
    metadata: Optional[Path] = None
    
    @validator('sra_list', 'fastq_dir')
    def check_input_source(cls, v, values):
        """Ensure either SRA list or FASTQ directory is provided."""
        if 'sra_list' in values and 'fastq_dir' in values:
            if bool(values.get('sra_list')) == bool(v):
                raise ValueError("Provide either SRA list or FASTQ directory, not both")
        return v


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    
    output_dir: Path
    keep_intermediate_files: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    
    # Core settings
    threads: int = Field(default=2, ge=1)
    memory_gb: int = Field(default=8, ge=1)
    
    # Configuration sections
    input: InputConfig
    output: OutputConfig
    genome: GenomeConfig = GenomeConfig()
    tools: ToolConfig = ToolConfig()
    
    # Advanced settings
    resume: bool = False
    dry_run: bool = False
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    @validator('input')
    def validate_input_config(cls, v):
        """Validate input configuration."""
        if not v.sra_list and not v.fastq_dir:
            raise ValueError("Either sra_list or fastq_dir must be provided")
        return v


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load and validate pipeline configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return PipelineConfig.from_yaml(config_path)


def create_default_config(output_path: Union[str, Path]) -> None:
    """Create a default configuration file."""
    default_config = {
        "threads": 2,
        "memory_gb": 8,
        "input": {
            "sra_list": None,
            "fastq_dir": None,
            "paired_end": False,
            "metadata": None
        },
        "output": {
            "output_dir": "./flync_results",
            "keep_intermediate_files": False,
            "log_level": "INFO"
        },
        "genome": {
            "species": "drosophila",
            "release": "BDGP6.32",
            "download_if_missing": True,
            "custom_genome": None,
            "custom_annotation": None
        },
        "tools": {
            "hisat2_threads": 1,
            "hisat2_extra_args": "",
            "stringtie_threads": 1,
            "stringtie_extra_args": "",
            "feature_extraction_threads": 1,
            "ml_model_path": None,
            "prediction_threshold": 0.5
        },
        "resume": False,
        "dry_run": False
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)
