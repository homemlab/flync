"""
Snakemake rules for transcriptome assembly with StringTie
"""

rule assemble_transcripts:
    """
    Assemble transcripts for each sample using StringTie
    """
    input:
        bam = OUTPUT_DIR / "data/{sample}/{sample}.sorted.bam",
        annotation = config["annotation"]
    output:
        gtf = OUTPUT_DIR / "assemblies/stringtie/{sample}.rna.gtf"
    params:
        stringtie_params = config.get("params", {}).get("stringtie_assemble", "")
    threads: config.get("threads", 8)
    log:
        OUTPUT_DIR / "logs/stringtie/{sample}.log"
    shell:
        """
        mkdir -p $(dirname {output.gtf})
        
        stringtie {input.bam} \
            -G {input.annotation} \
            -o {output.gtf} \
            -p {threads} \
            {params.stringtie_params} \
            &> {log}
        """
