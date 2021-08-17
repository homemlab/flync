#!/bin/bash

appdir=$1
cd $appdir
mkdir genome &> /dev/null
cd genome

### Download latest Drosophila melanogaster genome and annotation file ###

if [ -f 'dm6.fa' ]
then
    echo '!!!'WARNING'!!!' Genome sequence already downloaded. Skipping step.
else
    wget -q 'ftp://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz' -O dm6.fa.gz
    gzip -v -d --force dm6.fa.gz
fi

if [ -f 'dm6.gtf' ]
then
    echo '!!!'WARNING'!!!' Annotation already downloaded. Skipping step.
else
    ### Annotation files from different DBs ###
    ## Uncomment desired annotation source   ##
    wget -q 'ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ensGene.gtf.gz' -O dm6.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.ncbiRefSeq.gtf.gz" -O dm6.gtf.gz
    #wget -q "ftp://hgdownload.soe.ucsc.edu/goldenPath/dm6/bigZips/genes/dm6.refGene.gtf.gz" -O dm6.gtf.gz
    gzip -v -d --force dm6.gtf.gz
fi