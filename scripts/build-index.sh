#!/bin/bash

appdir=$1
threads=$2
cd $appdir
mkdir genome >&2 $workdir/err.log
cd genome
export HISAT2_INDEXES=$appdir/genome/

### Build genome index '('One-time-only command')'

idx=$(ls | grep -c '.ht2$')

if [ $idx == 0 ]
then
    echo ----- BUILDING GENOME INDEX -----
    hisat2-build -p $threads $appdir/genome/genome.fa genome.idx 2> idx.err.txt 1> idx.out.txt
    echo 'Done'

    echo ----- EXTRACTING SPLICE JUNCTIONS -----
    hisat2_extract_splice_sites.py $appdir/genome/genome.gtf > genome.sj
    echo 'Done'
fi
