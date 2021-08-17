#!/bin/bash

workdir=$1
cd $workdir/data

ls | grep $i'.hisat2.log$' >> logFiles.list

### Get stats from log files into a .tab file ###
echo -e sra_access'\t'num_reads'\t'percent_aligned > $workdir/results/map-stats.tsv
while read i
do
    a=$(ls | grep $i | awk -F '[.]' '{print $(NF-2)}')
    b=$(cat $i | sed 's@^[^0-9]*\([0-9]\+\).*@\1@' | grep -m1 '')
    c=$(cat $i | grep ' overall alignment rate' | awk -F '[%]' '{print $(NF-1)}')
    echo -e $a'\t'$b'\t'$c >> $workdir/results/map-stats.tsv
    rm $i
done < logFiles.list

rm logFiles.list
rm -rf $workdir/data/*.hisat2.log