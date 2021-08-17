#!/bin/bash
workdir=$1
sra=$2
cd $workdir
mkdir results &> /dev/null
echo sra_access,bio_proj > $workdir/results/runinfo.csv

while read -u 10 i; do
    esearch -db sra -query $i | efetch -format runinfo | cut -f1,22 -d, | grep SRR >> $workdir/results/runinfo.csv
done 10<$sra