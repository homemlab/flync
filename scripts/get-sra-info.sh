#!/bin/bash
workdir=$1
sra=$2
cd $workdir
mkdir results &> /dev/null
echo sra_access,avg_len,strategy,lib_prep,omic,read_layout,bio_proj > $workdir/results/runinfo.csv

echo ----- EXTRACTING RUN INFOS -----
while read -u 10 i; do
    esearch -db sra -query $i | efetch -format runinfo | cut -f1,7,13,14,15,16,22 -d, | cut -f2 -d$'\n' >> $workdir/results/runinfo.csv
done 10<$sra
echo 'Done'
