#!/bin/bash

workdir=$1
sra=$2
threads=$3
appdir=$4

cd $workdir

### Download and map reads from SRA accession numbers in .txt file ###

mkdir data
while read i
do
    mkdir data/$i
    echo ----- DOWNLOADING READS $i -----
    fastq-dump --gzip -O $workdir/data/$i $i

    echo ----- QC REPORT OF $i -----
    fastqc -t $threads $workdir/data/$i/$i'.fastq.gz'
    
    echo ----- MAPPING READS $i -----
    hisat2 -p $threads -x $appdir/genome/dm6.idx -U $workdir/data/$i/$i'.fastq.gz' -S $workdir/data/$i/$i'.sam' --dta --known-splicesite-infile $appdir/genome/dm6.ss &> $workdir/data/$i.hisat2.log

    echo ----- CONVERTING .SAM TO INDEXED SORTED .BAM OF $i -----
    samtools view -@ $threads -b -o $workdir/data/$i/$i'.bam' -S $workdir/data/$i/$i'.sam'
    samtools sort -@ $threads -o $workdir/data/$i/$i'.sorted.bam' $workdir/data/$i/$i'.bam' 
    samtools index $workdir/data/$i/$i'.sorted.bam'

    echo ----- REMOVING OBSOLETE READ FILES -----
    rm -rf $workdir/data/$i/$i'.fastq.gz'
    rm -rf $workdir/data/$i/$i'.sam'
    rm -rf $workdir/data/$i/$i'.bam'
done < $sra
