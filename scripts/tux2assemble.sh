#!/bin/bash

workdir=$1
sra=$2
threads=$3
appdir=$4
samthr=$(expr $threads / 2)
readlen=$(grep $2 $workdir/results/runinfo.csv | cut -f2 -d,)
sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
sd=${sd%.*}


cd $workdir

mkdir -p assemblies/stringtie
if [ ! -e $workdir/assemblies/stringtie/$2'.rna.gtf' ]; then
    echo ----- STRINGTIE TRANSCRIPTOME ASSEMBLY OF $2 -----
    stringtie $workdir/data/$2/$2'.sorted.bam' -G $appdir/genome/genome.gtf -o $workdir/assemblies/stringtie/$2'.rna.gtf'
    echo 'Done'
fi
wait
# if [ ! -e $workdir/assemblies/cufflinks/$2'_cuff'/transcripts.gtf ]; then
#     echo ----- CUFFLINKS TRANSCRIPTOME ASSEMBLY OF $2 -----
#     cufflinks --no-update-check -q -p $threads -o $workdir/assemblies/cufflinks/$2'_cuff' -g $appdir/genome/genome.gtf $workdir/data/$2/$2'.sorted.bam'
#     echo $workdir/assemblies/cufflinks/$2'_cuff'/transcripts.gtf >> $workdir/assemblies/cufflinks/cuffmerge.txt
#     echo 'Done'
# fi
