#!/bin/bash

appdir=$1
cd $appdir
mkdir genome &> /dev/null
cd genome

### Download latest Drosophila melanogaster genome and annotation file ###

if [ -f 'genome.fa' ]
then
    echo '!!!'WARNING'!!!' Genome sequence already downloaded. Skipping step.
else
    wget -q 'http://ftp.ensembl.org/pub/release-104/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.32.dna.toplevel.fa.gz' -O genome.fa.gz
    #wget -q 'ftp://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz' -O genome.fa.gz
    gzip -v -d --force genome.fa.gz
fi

if [ -f 'genome.gtf' ]
then
    echo '!!!'WARNING'!!!' Annotation already downloaded. Skipping step.
else
    ### Annotation files from different DBs ###
    ## Uncomment desired annotation source   ##
    wget -q 'http://ftp.ensembl.org/pub/release-104/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.32.104.chr.gtf.gz' -O genome.gtf.gz
    #wget -q 'ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ensGene.gtf.gz' -O genome.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ncbiRefSeq.gtf.gz" -O genome.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.refGene.gtf.gz" -O genome.gtf.gz
    gzip -v -d --force genome.gtf.gz
fi

if [ -f 'genome.cdna.fa' ]
then
    echo '!!!WARNING!!! Transcriptome already downloaded. Skipping step.'
else
    wget -q 'http://ftp.ensembl.org/pub/release-104/fasta/drosophila_melanogaster/cdna/Drosophila_melanogaster.BDGP6.32.cdna.all.fa.gz' -O genome.cdna.fa.gz
    gzip -v -d --force genome.cdna.fa.gz
fi

if ! [ $(grep --count ncRNA genome.gtf) == 0 ]; then
    grep 'transcript_biotype "ncRNA"' genome.gtf > genome.lncrna.gtf
    grep 'transcript_biotype "protein_coding"' genome.gtf > genome.cds.gtf
    grep -v 'transcript_biotype "ncRNA"' genome.gtf > genome.not.lncrna.gtf
    gffread -w genome.lncrna.fa -g genome.fa genome.lncrna.gtf
    gffread -w genome.cds.fa -g genome.fa genome.cds.gtf
fi

# Get only transcript and exon level features on gtf to ML-model
grep -v -P "\t"gene"\t" genome.lncrna.gtf > genome.lncrna.transOnly.gtf
cat genome.not.lncrna.gtf | grep -v -P "\t"gene"\t" | grep -v -P "\t"CDS"\t" | grep -v -P "\t"three_prime_utr"\t" | grep -v -P "\t"five_prime_utr"\t" | grep -v -P "\t"Selenocysteine"\t" > genome.not.lncrna.transOnly.gtf

bash $appdir/gtf-to-bed.sh genome.lncrna.transOnly.gtf
bash $appdir/gtf-to-bed.sh genome.not.lncrna.transOnly.gtf
