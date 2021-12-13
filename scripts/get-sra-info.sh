#!/bin/bash
workdir=$1
sra=$2
cd $workdir
mkdir results &> $workdir

echo ----- EXTRACTING RUN INFOS -----
if [ ! -e $workdir/results/runinfo.csv ]; then
    echo sra_access,avg_len,strategy,lib_prep,omic,read_layout,bio_proj > $workdir/results/runinfo.csv
    while read -u 10 i; do
        esearch -db sra -query $i | efetch -format runinfo | cut -f1,7,13,14,15,16,22 -d, | cut -f2 -d$'\n' >> $workdir/results/runinfo.csv
    done 10<$sra
fi
echo 'Done'
