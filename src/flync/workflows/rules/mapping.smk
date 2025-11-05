"""
Snakemake rules for read mapping with HISAT2

Supports two input modes:
1. SRA download mode (default): Downloads FASTQ from NCBI SRA
2. Local FASTQ mode: Uses existing FASTQ files from specified directory
"""

# Conditional rule execution based on fastq_dir config
USE_LOCAL_FASTQ = config.get("fastq_dir", None) is not None

if USE_LOCAL_FASTQ:
    # Local FASTQ mode: symlink or copy files to standard location
    rule prepare_local_fastq:
        """
        Symlink local FASTQ files to the expected output directory structure
        """
        output:
            fastq_se = OUTPUT_DIR / "data/{sample}/{sample}.fastq.gz",
            fastq_r1 = OUTPUT_DIR / "data/{sample}/{sample}_1.fastq.gz",
            fastq_r2 = OUTPUT_DIR / "data/{sample}/{sample}_2.fastq.gz"
        params:
            fastq_dir = config["fastq_dir"],
            paired = config.get("fastq_paired", False),
            outdir = lambda w: OUTPUT_DIR / f"data/{w.sample}",
            logdir = lambda w: OUTPUT_DIR / "logs/prepare_fastq",
            sample = "{sample}"
        log:
            OUTPUT_DIR / "logs/prepare_fastq/{sample}.log"
        shell:
            """
            mkdir -p {params.outdir}
            mkdir -p {params.logdir}
            
            if [ "{params.paired}" = "True" ]; then
                # Paired-end: look for _1 and _2 suffixes
                ln -sf {params.fastq_dir}/{params.sample}_1.fastq.gz {output.fastq_r1} 2> {log} || \
                ln -sf {params.fastq_dir}/{params.sample}_1.fq.gz {output.fastq_r1} 2>> {log} || \
                (echo "ERROR: Could not find {params.sample}_1.fastq.gz or _1.fq.gz in {params.fastq_dir}" >> {log} && exit 1)
                
                ln -sf {params.fastq_dir}/{params.sample}_2.fastq.gz {output.fastq_r2} 2>> {log} || \
                ln -sf {params.fastq_dir}/{params.sample}_2.fq.gz {output.fastq_r2} 2>> {log} || \
                (echo "ERROR: Could not find {params.sample}_2.fastq.gz or _2.fq.gz in {params.fastq_dir}" >> {log} && exit 1)
                
                # Create placeholder for single-end
                touch {output.fastq_se}
            else
                # Single-end: look for sample name without suffix
                ln -sf {params.fastq_dir}/{params.sample}.fastq.gz {output.fastq_se} 2> {log} || \
                ln -sf {params.fastq_dir}/{params.sample}.fq.gz {output.fastq_se} 2>> {log} || \
                (echo "ERROR: Could not find {params.sample}.fastq.gz or .fq.gz in {params.fastq_dir}" >> {log} && exit 1)
                
                # Create placeholders for paired-end
                touch {output.fastq_r1}
                touch {output.fastq_r2}
            fi
            """
else:
    # SRA download mode: fetch from NCBI
    rule download_sra:
        """
        Download reads from SRA using prefetch and fasterq-dump
        """
        output:
            fastq_se = OUTPUT_DIR / "data/{sample}/{sample}.fastq.gz",
            fastq_r1 = OUTPUT_DIR / "data/{sample}/{sample}_1.fastq.gz",
            fastq_r2 = OUTPUT_DIR / "data/{sample}/{sample}_2.fastq.gz"
        params:
            outdir = lambda w: OUTPUT_DIR / f"data/{w.sample}",
            logdir = lambda w: OUTPUT_DIR / "logs/download",
            sample = "{sample}",
            threads = config.get("params", {}).get("download_threads", 4)
        threads: config.get("params", {}).get("download_threads", 4)
        log:
            OUTPUT_DIR / "logs/download/{sample}.log"
        shell:
            """
            set -e  # Exit on error
            mkdir -p {params.outdir}
            mkdir -p {params.logdir}
            
            # Download SRA file (prefetch downloads to current dir or ncbi/public/sra/)
            echo "Starting prefetch for {params.sample}..." > {log}
            prefetch -O {params.outdir} {params.sample} 2>&1 | tee -a {log}
            
            # Convert to FASTQ with fasterq-dump
            echo "Converting to FASTQ..." >> {log}
            fasterq-dump -f -3 -e {threads} -O {params.outdir} {params.sample} 2>&1 | tee -a {log}
            
            # Compress FASTQ files if they exist
            if ls {params.outdir}/*.fastq 1> /dev/null 2>&1; then
                echo "Compressing FASTQ files..." >> {log}
                gzip {params.outdir}/*.fastq
            fi
            
            # Create placeholder files if not generated
            touch {output.fastq_se}
            touch {output.fastq_r1}
            touch {output.fastq_r2}
            
            echo "Download complete for {params.sample}" >> {log}
            """

rule map_reads:
    """
    Map reads to reference genome using HISAT2
    """
    input:
        fastq_se = OUTPUT_DIR / "data/{sample}/{sample}.fastq.gz",
        fastq_r1 = OUTPUT_DIR / "data/{sample}/{sample}_1.fastq.gz",
        fastq_r2 = OUTPUT_DIR / "data/{sample}/{sample}_2.fastq.gz",
        genome_index = config["hisat_index"] + ".1.ht2",
        splice_sites = config.get("splice_sites", config["genome"].replace(".fa", ".ss"))
    output:
        sam = temp(OUTPUT_DIR / "data/{sample}/{sample}.sam")
    params:
        index = config["hisat_index"],
        hisat_params = config.get("params", {}).get("hisat2", "-p 8"),
        sample = "{sample}"
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/hisat2/{sample}.log"
    shell:
        """
        # Use sample-specific temp directory to avoid FIFO name collisions
        export TMPDIR=$(mktemp -d -p /tmp hisat2_{wildcards.sample}_XXXX)
        trap "rm -rf $TMPDIR" EXIT
        
        # Determine if single-end or paired-end
        if [ -s {input.fastq_r1} ] && [ -s {input.fastq_r2} ]; then
            # Paired-end
            hisat2 {params.hisat_params} \
                -x {params.index} \
                -1 {input.fastq_r1} \
                -2 {input.fastq_r2} \
                -S {output.sam} \
                --dta --dta-cufflinks \
                --known-splicesite-infile {input.splice_sites} \
                &> {log}
        else
            # Single-end
            hisat2 {params.hisat_params} \
                -x {params.index} \
                -U {input.fastq_se} \
                -S {output.sam} \
                --dta --dta-cufflinks \
                --known-splicesite-infile {input.splice_sites} \
                &> {log}
        fi
        """

rule sam_to_sorted_bam:
    """
    Convert SAM to sorted BAM and create index
    """
    input:
        sam = OUTPUT_DIR / "data/{sample}/{sample}.sam"
    output:
        bam = OUTPUT_DIR / "data/{sample}/{sample}.sorted.bam",
        bai = OUTPUT_DIR / "data/{sample}/{sample}.sorted.bam.bai"
    params:
        temp_bam = lambda w: OUTPUT_DIR / f"data/{w.sample}/{w.sample}.bam"
    threads: config.get("threads", 8) // 2 if config.get("threads", 8) > 1 else 1
    log:
        OUTPUT_DIR / "logs/samtools/{sample}.log"
    shell:
        """
        # Convert SAM to BAM
        samtools view -@ {threads} -b -o {params.temp_bam} {input.sam} 2> {log}
        
        # Sort BAM
        samtools sort -@ {threads} -o {output.bam} {params.temp_bam} 2>> {log}
        
        # Index sorted BAM
        samtools index {output.bam} 2>> {log}
        
        # Clean up temporary BAM
        rm -f {params.temp_bam}
        """
