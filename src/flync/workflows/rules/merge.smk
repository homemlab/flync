"""
Snakemake rules for merging transcriptome assemblies
"""

rule create_merge_list:
    """
    Create a list of GTF files to merge
    """
    input:
        gtfs = expand(OUTPUT_DIR / "assemblies/stringtie/{sample}.rna.gtf", sample=SAMPLES)
    output:
        merge_list = OUTPUT_DIR / "assemblies/stringtie/gtf-to-merge.txt"
    shell:
        """
        ls {input.gtfs} > {output.merge_list}
        """

def get_batch_size(num_samples):
    """Calculate safe batch size based on number of samples"""
    # Conservative: aim for 400 GTFs per batch to avoid file handle limits
    if num_samples <= 500:
        return num_samples  # No batching needed
    return 400

checkpoint split_merge_list:
    """
    Split GTF list into manageable batches if needed
    """
    input:
        merge_list = OUTPUT_DIR / "assemblies/stringtie/gtf-to-merge.txt"
    output:
        batch_dir = directory(OUTPUT_DIR / "assemblies/stringtie/merge_batches")
    run:
        import math
        
        # Read all GTF files
        with open(input.merge_list) as f:
            gtf_files = [line.strip() for line in f if line.strip()]
        
        num_samples = len(gtf_files)
        batch_size = get_batch_size(num_samples)
        num_batches = math.ceil(num_samples / batch_size)
        
        # Create batch directory
        os.makedirs(output.batch_dir, exist_ok=True)
        
        # Write batch files
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            batch_gtfs = gtf_files[start_idx:end_idx]
            
            batch_file = os.path.join(output.batch_dir, f"batch_{i:03d}.txt")
            with open(batch_file, 'w') as f:
                f.write('\n'.join(batch_gtfs) + '\n')

def get_batch_files(wildcards):
    """Get list of batch files after checkpoint"""
    checkpoint_output = checkpoints.split_merge_list.get(**wildcards).output.batch_dir
    return expand(OUTPUT_DIR / "assemblies/stringtie/merge_batches/merged_batch_{i}.gtf",
                  i=glob_wildcards(os.path.join(checkpoint_output, "batch_{i}.txt")).i)

rule merge_batch:
    """
    Merge a single batch of GTF files
    """
    input:
        batch_list = OUTPUT_DIR / "assemblies/stringtie/merge_batches/batch_{batch_id}.txt",
        annotation = config["annotation"]
    output:
        batch_gtf = OUTPUT_DIR / "assemblies/stringtie/merge_batches/merged_batch_{batch_id}.gtf"
    params:
        merge_params = config.get("params", {}).get("stringtie_merge", "")
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/stringtie/merge_batch_{batch_id}.log"
    shell:
        """
        stringtie --merge {input.batch_list} \
            -G {input.annotation} \
            -o {output.batch_gtf} \
            -p {threads} \
            {params.merge_params} \
            &> {log}
        """

rule merge_transcripts:
    """
    Final merge of all batch assemblies into unified transcriptome
    """
    input:
        batch_gtfs = get_batch_files,
        annotation = config["annotation"]
    output:
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf",
        final_list = OUTPUT_DIR / "assemblies/stringtie/final-merge-list.txt"
    params:
        merge_params = config.get("params", {}).get("stringtie_merge", "")
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/stringtie/merge_final.log"
    shell:
        """
        mkdir -p $(dirname {output.merged_gtf})
        
        # Create list of batch merged GTFs
        ls {input.batch_gtfs} > {output.final_list}
        
        # Perform final merge
        stringtie --merge {output.final_list} \
            -G {input.annotation} \
            -o {output.merged_gtf} \
            -p {threads} \
            {params.merge_params} \
            &> {log}
        """

rule compare_assembly:
    """
    Compare merged assembly to reference annotation using gffcompare
    """
    input:
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf",
        annotation = config["annotation"]
    output:
        comparison = OUTPUT_DIR / "gffcompare/gffcmp.stats"
    params:
        outdir = OUTPUT_DIR / "gffcompare",
        prefix = OUTPUT_DIR / "gffcompare/gffcmp"
    log:
        OUTPUT_DIR / "logs/gffcompare/compare.log"
    shell:
        """
        mkdir -p {params.outdir}
        mkdir -p $(dirname {log})
        
        gffcompare -R \
            -r {input.annotation} \
            {input.merged_gtf} \
            -o {params.prefix} \
            &> {log}
        """

rule extract_all_transcripts:
    """
    Extract transcript sequences from merged assembly
    """
    input:
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf",
        genome = config["genome"]
    output:
        fasta = OUTPUT_DIR / "assemblies/assembled-transcripts.fa"
    log:
        OUTPUT_DIR / "logs/gffread/all_transcripts.log"
    shell:
        """
        gffread -w {output.fasta} \
            -g {input.genome} \
            {input.merged_gtf} \
            &> {log}
        """

rule filter_new_transcripts:
    """
    Filter merged GTF to keep only novel transcripts (MSTRG IDs)
    """
    input:
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf"
    output:
        filtered_gtf = OUTPUT_DIR / "assemblies/merged-new-transcripts.gtf"
    shell:
        """
        awk '$12 ~ /^"MSTRG*/' {input.merged_gtf} > {output.filtered_gtf}
        """

rule extract_new_transcripts:
    """
    Extract sequences for novel transcripts only
    """
    input:
        filtered_gtf = OUTPUT_DIR / "assemblies/merged-new-transcripts.gtf",
        genome = config["genome"]
    output:
        fasta = OUTPUT_DIR / "assemblies/assembled-new-transcripts.fa"
    log:
        OUTPUT_DIR / "logs/gffread/new_transcripts.log"
    shell:
        """
        gffread -w {output.fasta} \
            -g {input.genome} \
            {input.filtered_gtf} \
            &> {log}
        """
