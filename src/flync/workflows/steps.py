"""Pipeline step implementations."""

import shutil
from pathlib import Path
from typing import Dict, List

from ..pipeline import PipelineStep
from ..config import PipelineConfig


class GenomePreparationStep(PipelineStep):
    """Step 1: Prepare reference genome files."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "genome_preparation")
        self.genome_dir = config.output.output_dir / "genome"
    
    def validate_inputs(self) -> bool:
        """Validate genome preparation inputs."""
        # Check if custom genome files are provided and exist
        if self.config.genome.custom_genome:
            if not self.config.genome.custom_genome.exists():
                self.logger.error(f"Custom genome file not found: {self.config.genome.custom_genome}")
                return False
        
        if self.config.genome.custom_annotation:
            if not self.config.genome.custom_annotation.exists():
                self.logger.error(f"Custom annotation file not found: {self.config.genome.custom_annotation}")
                return False
        
        return True
    
    def run(self) -> bool:
        """Download or prepare genome files."""
        self.genome_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if genome files already exist
        genome_fa = self.genome_dir / "genome.fa"
        genome_gtf = self.genome_dir / "genome.gtf"
        
        if genome_fa.exists() and genome_gtf.exists():
            self.logger.info("Genome files already exist, skipping download")
            return True
        
        if self.config.genome.custom_genome and self.config.genome.custom_annotation:            # Use custom genome files
            self.logger.info("Using custom genome files")
            shutil.copy2(self.config.genome.custom_genome, genome_fa)
            shutil.copy2(self.config.genome.custom_annotation, genome_gtf)
        else:
            # Download default genome
            return self._download_default_genome()
        
        return True
    
    def _download_default_genome(self) -> bool:
        """Download the default Drosophila genome."""
        self.logger.info(f"Downloading {self.config.genome.species} genome release {self.config.genome.release}")
        
        # URLs for Drosophila BDGP6.32 release 106 (matching original script)
        base_url = "https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/"
        chromosomes = ["2L", "2R", "3L", "3R", "4", "X", "Y", "mitochondrion_genome"]
        
        downloaded_files = []
        
        # Download chromosome files
        for chrom in chromosomes:
            if chrom == "mitochondrion_genome":
                filename = f"Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.{chrom}.fa.gz"
            else:
                filename = f"Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.{chrom}.fa.gz"
            
            url = base_url + filename
            local_file = self.genome_dir / f"genome.{chrom}.fa.gz"
            
            self.logger.info(f"Downloading {chrom} chromosome")
            result = self.run_command([
                "wget", "-q", url, "-O", str(local_file)
            ])
            
            if result.returncode != 0:
                self.logger.error(f"Failed to download {chrom} chromosome")
                return False
            
            downloaded_files.append(local_file)
        
        # Concatenate all chromosome files
        self.logger.info("Concatenating chromosome files")
        genome_fa = self.genome_dir / "genome.fa"
        
        # Use zcat to concatenate gzipped files
        with open(genome_fa, 'w') as outfile:
            for gz_file in downloaded_files:
                result = self.run_command(["zcat", str(gz_file)], capture_output=True)
                if result.returncode == 0:
                    outfile.write(result.stdout)
                else:
                    self.logger.error(f"Failed to decompress {gz_file}")
                    return False
        
        # Clean up individual chromosome files
        for gz_file in downloaded_files:
            gz_file.unlink()
        
        # Download annotation
        gtf_url = "https://ftp.ensembl.org/pub/release-106/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.32.106.chr.gtf.gz"
        gtf_gz = self.genome_dir / "genome.gtf.gz"
        
        self.logger.info("Downloading genome annotation")
        result = self.run_command([
            "wget", "-q", gtf_url, "-O", str(gtf_gz)
        ])
        
        if result.returncode != 0:
            self.logger.error("Failed to download annotation")
            return False
        
        # Decompress annotation
        self.run_command(["gunzip", str(gtf_gz)])
        
        self.logger.info("Genome download completed successfully")
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get genome preparation outputs."""
        return {
            "genome_fasta": self.genome_dir / "genome.fa",
            "genome_gtf": self.genome_dir / "genome.gtf",
            "genome_dir": self.genome_dir
        }


class InputPreparationStep(PipelineStep):
    """Step 2: Prepare input data (SRA info or FASTQ validation)."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "input_preparation")
        self.output_dir = config.output.output_dir
    
    def validate_inputs(self) -> bool:
        """Validate input preparation."""
        if self.config.input.sra_list:
            if not self.config.input.sra_list.exists():
                self.logger.error(f"SRA list file not found: {self.config.input.sra_list}")
                return False
        elif self.config.input.fastq_dir:
            if not self.config.input.fastq_dir.exists():
                self.logger.error(f"FASTQ directory not found: {self.config.input.fastq_dir}")
                return False
        else:
            self.logger.error("Neither SRA list nor FASTQ directory provided")
            return False
        
        return True
    
    def run(self) -> bool:
        """Prepare input data."""
        if self.config.input.sra_list:
            return self._prepare_sra_input()
        else:
            return self._prepare_fastq_input()
    
    def _prepare_sra_input(self) -> bool:
        """Prepare SRA input data."""
        self.logger.info("Preparing SRA input data")
        
        # Validate SRA accessions
        with open(self.config.input.sra_list, 'r') as f:
            sra_accessions = [line.strip() for line in f if line.strip()]
        
        self.logger.info(f"Found {len(sra_accessions)} SRA accessions")
        
        # Create SRA info file for downstream processing
        sra_info_file = self.output_dir / "sra_info.txt"
        with open(sra_info_file, 'w') as f:
            for acc in sra_accessions:
                f.write(f"{acc}\\n")
        
        return True
    
    def _prepare_fastq_input(self) -> bool:
        """Prepare FASTQ input data."""
        self.logger.info("Preparing FASTQ input data")
        
        # Find FASTQ files
        if self.config.input.paired_end:
            r1_files = list(self.config.input.fastq_dir.glob("*_1.fastq.gz"))
            r2_files = list(self.config.input.fastq_dir.glob("*_2.fastq.gz"))
            
            if not r1_files:
                r1_files = list(self.config.input.fastq_dir.glob("*_R1.fastq.gz"))
                r2_files = list(self.config.input.fastq_dir.glob("*_R2.fastq.gz"))
            
            self.logger.info(f"Found {len(r1_files)} paired-end FASTQ file pairs")
            
            if len(r1_files) != len(r2_files):
                self.logger.error("Mismatch in paired-end FASTQ files")
                return False
                
        else:
            fastq_files = list(self.config.input.fastq_dir.glob("*.fastq.gz"))
            self.logger.info(f"Found {len(fastq_files)} single-end FASTQ files")
            
            if not fastq_files:
                self.logger.error("No FASTQ files found")
                return False
        
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get input preparation outputs."""
        outputs = {}
        
        if self.config.input.sra_list:
            outputs["sra_info"] = self.output_dir / "sra_info.txt"
        else:
            outputs["fastq_dir"] = self.config.input.fastq_dir
            
        return outputs


class MappingStep(PipelineStep):
    """Step 3: Read mapping using HISAT2."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "mapping")
        self.mapped_dir = config.output.output_dir / "mapped"
        self.hisat2_index_dir = config.output.output_dir / "genome" / "hisat2_index"
    
    def validate_inputs(self) -> bool:
        """Validate mapping inputs."""
        genome_dir = self.config.output.output_dir / "genome"
        if not (genome_dir / "genome.fa").exists():
            self.logger.error("Genome FASTA file not found")
            return False
        
        return True
    
    def run(self) -> bool:
        """Run read mapping."""
        self.mapped_dir.mkdir(parents=True, exist_ok=True)
        
        # Build HISAT2 index if it doesn't exist
        if not self._build_hisat2_index():
            return False
        
        # Map reads
        if self.config.input.sra_list:
            return self._map_sra_reads()
        else:
            return self._map_fastq_reads()
      def _build_hisat2_index(self) -> bool:
        """Build HISAT2 index."""
        self.hisat2_index_dir.mkdir(parents=True, exist_ok=True)
        index_base = self.hisat2_index_dir / "genome.idx"
        
        # Check if index already exists
        if list(self.hisat2_index_dir.glob("*.ht2")):
            self.logger.info("HISAT2 index already exists")
            return True
        
        self.logger.info("Building HISAT2 index")
        genome_fa = self.config.output.output_dir / "genome" / "genome.fa"
        genome_gtf = self.config.output.output_dir / "genome" / "genome.gtf"
        
        # Build main index
        cmd = [
            "hisat2-build",
            "-p", str(self.config.tools.hisat2_threads),
            str(genome_fa),
            str(index_base)
        ]
        
        result = self.run_command(cmd)
        if result.returncode != 0:
            self.logger.error("Failed to build HISAT2 index")
            return False
        
        # Extract splice sites
        self.logger.info("Extracting splice sites")
        splice_sites_file = self.hisat2_index_dir / "genome.ss"
        cmd = [
            "hisat2_extract_splice_sites.py",
            str(genome_gtf)
        ]
        
        result = self.run_command(cmd, capture_output=True)
        if result.returncode == 0:
            with open(splice_sites_file, 'w') as f:
                f.write(result.stdout)
        else:
            self.logger.warning("Failed to extract splice sites, continuing without them")
        
        return True
    
    def _map_sra_reads(self) -> bool:
        """Map SRA reads (download and map)."""
        self.logger.info("Mapping SRA reads")
        # TODO: Implement SRA read mapping
        # This would involve downloading SRA files and mapping them
        return True
    
    def _map_fastq_reads(self) -> bool:
        """Map FASTQ reads."""
        self.logger.info("Mapping FASTQ reads")
        # TODO: Implement FASTQ read mapping
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get mapping outputs."""
        return {
            "mapped_dir": self.mapped_dir,
            "hisat2_index": self.hisat2_index_dir
        }


class AssemblyStep(PipelineStep):
    """Step 4: Transcriptome assembly using StringTie."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "assembly")
        self.assembly_dir = config.output.output_dir / "assembled"
    
    def validate_inputs(self) -> bool:
        """Validate assembly inputs."""
        mapped_dir = self.config.output.output_dir / "mapped"
        if not mapped_dir.exists():
            self.logger.error("Mapped reads directory not found")
            return False
        return True
    
    def run(self) -> bool:
        """Run transcriptome assembly."""
        self.assembly_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Running StringTie assembly")
        # TODO: Implement StringTie assembly
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get assembly outputs."""
        return {
            "assembly_dir": self.assembly_dir,
            "merged_gtf": self.assembly_dir / "merged.gtf"
        }


class CodingProbabilityStep(PipelineStep):
    """Step 5: Calculate coding probability and classify transcripts."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "coding_probability")
        self.results_dir = config.output.output_dir / "results"
    
    def validate_inputs(self) -> bool:
        """Validate coding probability inputs."""
        assembly_dir = self.config.output.output_dir / "assembled"
        if not assembly_dir.exists():
            self.logger.error("Assembly directory not found")
            return False
        return True
    
    def run(self) -> bool:
        """Calculate coding probability."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Calculating coding probability")
        # TODO: Implement coding probability calculation
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get coding probability outputs."""
        return {
            "results_dir": self.results_dir,
            "non_coding_gtf": self.results_dir / "new-non-coding.gtf"
        }


class DifferentialExpressionStep(PipelineStep):
    """Step 6: Differential expression analysis (optional)."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "differential_expression")
        self.results_dir = config.output.output_dir / "results"
    
    def validate_inputs(self) -> bool:
        """Validate differential expression inputs."""
        if not self.config.input.metadata:
            return False
            
        if not self.config.input.metadata.exists():
            self.logger.error(f"Metadata file not found: {self.config.input.metadata}")
            return False
            
        return True
    
    def run(self) -> bool:
        """Run differential expression analysis."""
        self.logger.info("Running differential expression analysis")
        # TODO: Implement differential expression analysis
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get differential expression outputs."""
        return {
            "de_results": self.results_dir / "ballgown_results.csv"
        }


class FeatureExtractionStep(PipelineStep):
    """Step 7: Extract genomic features."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "feature_extraction")
        self.features_dir = config.output.output_dir / "results" / "non-coding" / "features"
    
    def validate_inputs(self) -> bool:
        """Validate feature extraction inputs."""
        non_coding_gtf = self.config.output.output_dir / "results" / "new-non-coding.gtf"
        if not non_coding_gtf.exists():
            self.logger.error("Non-coding GTF file not found")
            return False
        return True
    
    def run(self) -> bool:
        """Extract genomic features."""
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Extracting genomic features")
        # TODO: Implement feature extraction
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get feature extraction outputs."""
        return {
            "features_dir": self.features_dir,
            "feature_table": self.config.output.output_dir / "results" / "new-non-coding.csv"
        }


class PredictionStep(PipelineStep):
    """Step 8: ML prediction of non-coding RNAs."""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config, "prediction")
        self.results_dir = config.output.output_dir / "results"
    
    def validate_inputs(self) -> bool:
        """Validate prediction inputs."""
        feature_table = self.results_dir / "new-non-coding.csv"
        if not feature_table.exists():
            self.logger.error("Feature table not found")
            return False
        return True
    
    def run(self) -> bool:
        """Run ML prediction."""
        self.logger.info("Running ML prediction")
        # TODO: Implement ML prediction
        return True
    
    def get_outputs(self) -> Dict[str, Path]:
        """Get prediction outputs."""
        return {
            "predictions": self.results_dir / "final_predictions.csv",
            "model_output": self.results_dir / "model_predictions.csv"
        }
