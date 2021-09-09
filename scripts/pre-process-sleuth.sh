#!/bin/bash

workdir=$1
sra=$2
metadata=$3

cd $workdir
mkdir results_dea &> /dev/null
cd results_dea

### Create a .csv file for sleuth analysis ###
echo sample,condition,path > $workdir/results_dea/metadata.csv
while read i
do
    srr=$(cat $metadata | grep $i | cut -f1 -d,)
    srrcond=$(cat $metadata | grep $i | cut -f2 -d,)
    
    if [ $( echo $srr | wc -l ) == 1 ]; then
        echo $srr','$srrcond','$workdir/kallisto/$srr >> $workdir/results_dea/metadata.csv
    fi
done < $sra
