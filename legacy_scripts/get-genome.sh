#!/usr/bin/env bash

appdir=$1
cd $appdir
mkdir -p genome 
cd genome

### Download Drosophila melanogaster BDGP6.32 release 106 genome and annotation file ###

if [ -f 'genome.fa' ]
then
    echo '!!!'WARNING'!!!' Genome sequence already downloaded. Skipping step.
else
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.2L.fa.gz' -O genome.2L.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.2R.fa.gz' -O genome.2R.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.3L.fa.gz' -O genome.3L.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.3R.fa.gz' -O genome.3R.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.4.fa.gz' -O genome.4.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.X.fa.gz' -O genome.X.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.Y.fa.gz' -O genome.Y.fa.gz
    wget -q 'https://ftp.ensembl.org/pub/release-106/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.primary_assembly.mitochondrion_genome.fa.gz' -O genome.mitochondrion_genome.fa.gz
    zcat genome.2L.fa.gz genome.2R.fa.gz genome.3L.fa.gz genome.3R.fa.gz genome.4.fa.gz genome.X.fa.gz genome.Y.fa.gz genome.mitochondrion_genome.fa.gz > genome.fa    
    rm genome.*.fa.gz   
fi

if [ -f 'genome.gtf' ]
then
    echo '!!!'WARNING'!!!' Annotation already downloaded. Skipping step.
else
    ### Annotation files ###
    wget -q 'https://ftp.ensembl.org/pub/release-106/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.32.106.chr.gtf.gz' -O genome.gtf.gz
    gzip -v -d --force genome.gtf.gz
fi

# if [ -f 'genome.cdna.fa' ]
# then
#     echo '!!!WARNING!!! Transcriptome already downloaded. Skipping step.'
# else
#     wget -q 'http://ftp.ensembl.org/pub/release-104/fasta/drosophila_melanogaster/cdna/Drosophila_melanogaster.BDGP6.32.cdna.all.fa.gz' -O genome.cdna.fa.gz
#     gzip -v -d --force genome.cdna.fa.gz
# fi

# if ! [ $(grep --count ncRNA genome.gtf) == 0 ]; then
#     grep 'transcript_biotype "ncRNA"' genome.gtf > genome.lncrna.gtf
#     grep 'transcript_biotype "protein_coding"' genome.gtf > genome.cds.gtf
#     grep -v 'transcript_biotype "ncRNA"' genome.gtf > genome.not.lncrna.gtf
#     gffread -w genome.lncrna.fa -g genome.fa genome.lncrna.gtf
#     gffread -w genome.cds.fa -g genome.fa genome.cds.gtf
# fi

# Get only transcript and exon level features on gtf to ML-model
# grep -v -P "\t"gene"\t" genome.lncrna.gtf > genome.lncrna.transOnly.gtf
# cat genome.not.lncrna.gtf | grep -v -P "\t"gene"\t" | grep -v -P "\t"CDS"\t" | grep -v -P "\t"three_prime_utr"\t" | grep -v -P "\t"five_prime_utr"\t" | grep -v -P "\t"Selenocysteine"\t" | grep -v -P "\t"start_codon"\t" | grep -v -P "\t"stop_codon"\t" > genome.not.lncrna.transOnly.gtf

# bash $appdir/scripts/gtf-to-bed.sh genome.lncrna.transOnly.gtf
# bash $appdir/scripts/gtf-to-bed.sh genome.not.lncrna.transOnly.gtf
