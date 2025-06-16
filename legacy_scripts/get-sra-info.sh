#!/usr/bin/env bash

workdir=$1
sra=$2
cd $workdir
mkdir -p results 

echo ----- EXTRACTING RUN INFOS -----
if [ ! -e $workdir/results/runinfo.csv ]; then
    touch $workdir/results/runinfo.csv
    echo sra_access,avg_len,strategy,lib_prep,omic,read_layout,bio_proj > $workdir/results/runinfo.csv
    
     while read i; do 
        efetch -db sra -format runinfo -id $i | cut -f1,7,13,14,15,16,22 -d, | egrep -v "^Run," >> $workdir/results/runinfo.csv
     done < $sra

fi
echo 'Done'
