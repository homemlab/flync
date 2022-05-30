#!/usr/bin/env bash

workdir=$1
sra=$2

cd $workdir

### Re-run assmebler to output gene counts
mkdir -p $workdir/cov
if [ ! -e $workdir/cov/$sra'.rna.gtf' ]; then
    echo ----- STRINGTIE TRANSCRIPT COUNTS OF $sra -----
    stringtie -eB $workdir/data/$sra/$sra'.sorted.bam' -G $workdir/assemblies/merged.gtf -o $workdir/cov/$sra/$sra'.rna.gtf'
    echo 'Done'
fi


