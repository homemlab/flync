"""
Snakemake rules for transcript quantification
"""

rule quantify_transcripts:
    """
    Re-quantify transcripts using merged assembly as reference
    """
    input:
        bam = OUTPUT_DIR / "data/{sample}/{sample}.sorted.bam",
        merged_gtf = OUTPUT_DIR / "assemblies/merged.gtf"
    output:
        gtf = OUTPUT_DIR / "cov/{sample}/{sample}.rna.gtf",
        coverage_dir = directory(OUTPUT_DIR / "cov/{sample}")
    params:
        quantify_params = config.get("params", {}).get("stringtie_quantify", "-eB")
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/quantify/{sample}.log"
    shell:
        """
        mkdir -p {output.coverage_dir}
        
        stringtie {params.quantify_params} \
            {input.bam} \
            -G {input.merged_gtf} \
            -o {output.gtf} \
            -p {threads} \
            &> {log}
        """

rule aggregate_counts:
    """
    Create completion marker after all quantifications finish
    """
    input:
        gtfs = expand(OUTPUT_DIR / "cov/{sample}/{sample}.rna.gtf", sample=SAMPLES)
    output:
        marker = OUTPUT_DIR / "cov/quantification_complete.txt"
    shell:
        """
        echo "Quantification completed for all samples" > {output.marker}
        """
