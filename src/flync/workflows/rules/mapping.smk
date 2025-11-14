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
        Auto-detects library layout from file naming patterns
        """
        output:
            fastq_se = OUTPUT_DIR / "data/{sample}/{sample}.fastq.gz",
            fastq_r1 = OUTPUT_DIR / "data/{sample}/{sample}_1.fastq.gz",
            fastq_r2 = OUTPUT_DIR / "data/{sample}/{sample}_2.fastq.gz"
        params:
            fastq_dir = config["fastq_dir"],
            outdir = lambda w: OUTPUT_DIR / f"data/{w.sample}",
            logdir = lambda w: OUTPUT_DIR / "logs/prepare_fastq",
            sample = "{sample}",
            layout_file = LAYOUT_STORAGE_FILE,
            script_dir = Path(workflow.basedir) / "scripts"
        log:
            OUTPUT_DIR / "logs/prepare_fastq/{sample}.log"
        shell:
            """
            mkdir -p {params.outdir}
            mkdir -p {params.logdir}
            
            # Check if layout is already known (from mapping file or global config)
            EXPECTED_LAYOUT=$(python3 {params.script_dir}/library_layout_manager.py \\
                {params.layout_file} get {params.sample} 2>/dev/null || echo "UNKNOWN")
            
            # Try to find files and auto-detect if layout unknown
            FOUND_PAIRED="false"
            FOUND_SINGLE="false"
            
            # Check for paired-end files
            if [ -f "{params.fastq_dir}/{params.sample}_1.fastq.gz" ] && [ -f "{params.fastq_dir}/{params.sample}_2.fastq.gz" ]; then
                FOUND_PAIRED="true"
            elif [ -f "{params.fastq_dir}/{params.sample}_1.fq.gz" ] && [ -f "{params.fastq_dir}/{params.sample}_2.fq.gz" ]; then
                FOUND_PAIRED="true"
            fi
            
            # Check for single-end files
            if [ -f "{params.fastq_dir}/{params.sample}.fastq.gz" ]; then
                FOUND_SINGLE="true"
            elif [ -f "{params.fastq_dir}/{params.sample}.fq.gz" ]; then
                FOUND_SINGLE="true"
            fi
            
            # Determine actual layout
            if [ "$FOUND_PAIRED" = "true" ]; then
                DETECTED_LAYOUT="PAIRED"
            elif [ "$FOUND_SINGLE" = "true" ]; then
                DETECTED_LAYOUT="SINGLE"
            else
                echo "ERROR: Could not find FASTQ files for {params.sample} in {params.fastq_dir}" >> {log}
                echo "  Looked for:" >> {log}
                echo "    Paired: {params.sample}_1.fastq.gz + {params.sample}_2.fastq.gz" >> {log}
                echo "    Single: {params.sample}.fastq.gz" >> {log}
                exit 1
            fi
            
            echo "Detected layout for {params.sample}: $DETECTED_LAYOUT" > {log}
            
            # Store detected layout
            python3 {params.script_dir}/library_layout_manager.py \\
                {params.layout_file} set {params.sample} \\
                $([ "$DETECTED_LAYOUT" = "PAIRED" ] && echo "true" || echo "false") \\
                >> {log} 2>&1
            
            # Validate if expected layout was set
            if [ "$EXPECTED_LAYOUT" != "UNKNOWN" ] && [ "$EXPECTED_LAYOUT" != "$DETECTED_LAYOUT" ]; then
                echo "========================================" >> {log}
                echo "ERROR: Library layout mismatch" >> {log}
                echo "========================================" >> {log}
                echo "Expected: $EXPECTED_LAYOUT" >> {log}
                echo "Found: $DETECTED_LAYOUT" >> {log}
                echo "Sample: {params.sample}" >> {log}
                echo "========================================" >> {log}
                exit 1
            fi
            
            # Symlink files based on detected layout
            if [ "$DETECTED_LAYOUT" = "PAIRED" ]; then
                # Paired-end: symlink _1 and _2
                ln -sf {params.fastq_dir}/{params.sample}_1.fastq.gz {output.fastq_r1} 2>> {log} || \\
                ln -sf {params.fastq_dir}/{params.sample}_1.fq.gz {output.fastq_r1} 2>> {log}
                
                ln -sf {params.fastq_dir}/{params.sample}_2.fastq.gz {output.fastq_r2} 2>> {log} || \\
                ln -sf {params.fastq_dir}/{params.sample}_2.fq.gz {output.fastq_r2} 2>> {log}
                
                touch {output.fastq_se}
                echo "Symlinked paired-end files" >> {log}
            else
                # Single-end: symlink single file
                ln -sf {params.fastq_dir}/{params.sample}.fastq.gz {output.fastq_se} 2>> {log} || \\
                ln -sf {params.fastq_dir}/{params.sample}.fq.gz {output.fastq_se} 2>> {log}
                
                touch {output.fastq_r1}
                touch {output.fastq_r2}
                echo "Symlinked single-end file" >> {log}
            fi
            
            echo "Preparation complete for {params.sample}" >> {log}
            """
else:
    # SRA download mode: fetch from NCBI
    rule download_sra:
        """
        Download reads from SRA using prefetch and fasterq-dump
        Automatically detects and validates library layout per sample
        """
        output:
            fastq_se = OUTPUT_DIR / "data/{sample}/{sample}.fastq.gz",
            fastq_r1 = OUTPUT_DIR / "data/{sample}/{sample}_1.fastq.gz",
            fastq_r2 = OUTPUT_DIR / "data/{sample}/{sample}_2.fastq.gz"
        params:
            outdir = lambda w: OUTPUT_DIR / f"data/{w.sample}",
            logdir = lambda w: OUTPUT_DIR / "logs/download",
            sample = "{sample}",
            threads = config.get("params", {}).get("download_threads", 4),
            layout_file = LAYOUT_STORAGE_FILE,
            script_dir = Path(workflow.basedir) / "scripts"
        threads: config.get("params", {}).get("download_threads", 4)
        log:
            OUTPUT_DIR / "logs/download/{sample}.log"
        shell:
            """
            set -e  # Exit on error
            mkdir -p {params.outdir}
            mkdir -p {params.logdir}
            
            # Download SRA file
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
            
            # Detect actual library type from downloaded files
            SE_FILE="{params.outdir}/{params.sample}.fastq.gz"
            R1_FILE="{params.outdir}/{params.sample}_1.fastq.gz"
            R2_FILE="{params.outdir}/{params.sample}_2.fastq.gz"
            
            HAS_PAIRED="false"
            HAS_SINGLE="false"
            
            if [ -f "$R1_FILE" ] && [ -f "$R2_FILE" ] && [ -s "$R1_FILE" ] && [ -s "$R2_FILE" ]; then
                HAS_PAIRED="true"
                DETECTED_LAYOUT="PAIRED"
                echo "Detected PAIRED-END reads" >> {log}
            fi
            
            if [ -f "$SE_FILE" ] && [ -s "$SE_FILE" ]; then
                HAS_SINGLE="true"
                DETECTED_LAYOUT="SINGLE"
                echo "Detected SINGLE-END reads" >> {log}
            fi
            
            # Store detected layout
            python3 {params.script_dir}/library_layout_manager.py \\
                {params.layout_file} set {params.sample} \\
                $([ "$HAS_PAIRED" = "true" ] && echo "true" || echo "false") \\
                >> {log} 2>&1
            
            # Check if layout was expected (from mapping file or global config)
            EXPECTED_LAYOUT=$(python3 {params.script_dir}/library_layout_manager.py \\
                {params.layout_file} get {params.sample} 2>/dev/null || echo "UNKNOWN")
            
            if [ "$EXPECTED_LAYOUT" != "UNKNOWN" ] && [ "$EXPECTED_LAYOUT" != "$DETECTED_LAYOUT" ]; then
                echo "========================================" >> {log}
                echo "ERROR: Library layout mismatch detected" >> {log}
                echo "========================================" >> {log}
                echo "" >> {log}
                echo "Expected: $EXPECTED_LAYOUT" >> {log}
                echo "Detected: $DETECTED_LAYOUT" >> {log}
                echo "Sample: {params.sample}" >> {log}
                echo "" >> {log}
                echo "SOLUTION: Update your library_layout_file or config.yml" >> {log}
                echo "========================================" >> {log}
                exit 1
            fi
            
            # Create placeholder files if not generated
            touch {output.fastq_se}
            touch {output.fastq_r1}
            touch {output.fastq_r2}
            
            echo "Download complete for {params.sample}" >> {log}
            echo "Library layout: $DETECTED_LAYOUT" >> {log}
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
