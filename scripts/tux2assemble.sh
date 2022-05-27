#!/usr/bin/env bash

workdir=$1
sra=$2
threads=$3
appdir=$4

cd $workdir

mkdir -p assemblies/stringtie
if [ ! -e $workdir/assemblies/stringtie/$sra'.rna.gtf' ]; then
    echo ----- STRINGTIE TRANSCRIPTOME ASSEMBLY OF $sra -----
    stringtie $workdir/data/$sra/$sra'.sorted.bam' -G $appdir/genome/genome.gtf -o $workdir/assemblies/stringtie/$sra'.rna.gtf'
    echo 'Done'
fi
wait
# if [ ! -e $workdir/assemblies/cufflinks/$sra'_cuff'/transcripts.gtf ]; then
#     echo ----- CUFFLINKS TRANSCRIPTOME ASSEMBLY OF $sra -----
#     cufflinks --no-update-check -q -p $threads -o $workdir/assemblies/cufflinks/$sra'_cuff' -g $appdir/genome/genome.gtf $workdir/data/$sra/$sra'.sorted.bam'
#     echo $workdir/assemblies/cufflinks/$sra'_cuff'/transcripts.gtf >> $workdir/assemblies/cufflinks/cuffmerge.txt
#     echo 'Done'
# fi
