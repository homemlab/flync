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

rule merge_transcripts:
    """
    Merge all sample assemblies into a unified transcriptome
    """
    input:
        merge_list = OUTPUT_DIR / "assemblies/stringtie/gtf-to-merge.txt",
        annotation = config["annotation"]
    output:
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf"
    params:
        merge_params = config.get("params", {}).get("stringtie_merge", "")
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/stringtie/merge.log"
    shell:
        """
        mkdir -p $(dirname {output.merged_gtf})
        
        stringtie --merge {input.merge_list} \
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
