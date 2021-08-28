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
    wget -q 'http://ftp.ensembl.org/pub/release-104/gtf/drosophila_melanogaster/Drosophila_melanogaster.BDGP6.32.104.gtf.gz' -O genome.gtf.gz
    #wget -q 'ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ensGene.gtf.gz' -O genome.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ncbiRefSeq.gtf.gz" -O genome.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.refGene.gtf.gz" -O genome.gtf.gz
    gzip -v -d --force genome.gtf.gz
fi

if ! [ $(grep --count ncRNA genome.gtf) == 0 ]; then
    grep ncRNA genome.gtf > genome.ncrna.gtf
    grep protein_coding.gtf > genome.cds.gtf
fi

if ! [ -f 'genome.ncrna.fa' ]
then
    wget -q 'http://ftp.ensembl.org/pub/release-104/fasta/drosophila_melanogaster/ncrna/Drosophila_melanogaster.BDGP6.32.ncrna.fa.gz' -O genome.ncrna.fa.gz
    gzip -v -d --force genome.ncrna.fa.gz
fi

if ! [ -f 'genome.cds.fa' ]
then
    wget -q 'http://ftp.ensembl.org/pub/release-104/fasta/drosophila_melanogaster/cds/Drosophila_melanogaster.BDGP6.32.cds.all.fa.gz' -O genome.cds.fa.gz
    gzip -v -d --force genome.cds.fa.gz
fi
