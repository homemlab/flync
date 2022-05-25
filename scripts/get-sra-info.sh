#!/usr/bin/env bash

workdir=$1
sra=$2
cd $workdir
mkdir -p results 

echo ----- EXTRACTING RUN INFOS -----
if [ ! -e $workdir/results/runinfo.csv ]; then
    touch $workdir/results/runinfo.csv
    echo sra_access,avg_len,strategy,lib_prep,omic,read_layout,bio_proj > $workdir/results/runinfo.csv
        while read -u 10 i; do
        esearch -db sra -query $i | efetch -format runinfo | cut -f1,7,13,14,15,16,22 -d',' >> $workdir/results/preruninfo.csv
    done 10<$sra
    cat $workdir/results/preruninfo.csv | sort | uniq > $workdir/results/runinfo.csv
    rm $workdir/results/preruninfo.csv
fi
echo 'Done'
