#!/bin/bash

workdir=$1
sra=$2
threads=$3
appdir=$4

cd $workdir

### Download and map reads from SRA accession numbers in .txt file ###

mkdir data &> /dev/null
while read i
do
    mkdir data/$i &> /dev/null
    echo ----- DOWNLOADING READS $i -----
    ## Faster:
    fasterq-dump -f -3 -p -e $threads -O $workdir/data/$i $i
    gzip *.fastq
    ## Old command:
    #fastq-dump --gzip -O $workdir/data/$i $i 2> $workdir/data/$i/fastq.out.txt 1> /dev/null
    echo 'Done'

    echo ----- QC REPORT OF $i -----
    fastqc -t $threads $workdir/data/$i/$i'.fastq.gz' 2> $workdir/data/$i/fastq.out.txt 1> /dev/null
    echo 'Done'    
    
    echo ----- MAPPING READS $i -----
    hisat2 -p $threads -x $appdir/genome/genome.idx -U $workdir/data/$i/$i'.fastq.gz' -S $workdir/data/$i/$i'.sam' --dta --known-splicesite-infile $appdir/genome/genome.sj &> $workdir/data/$i.hisat2.log
    echo 'Done'

    echo ----- CONVERTING .SAM TO INDEXED SORTED .BAM OF $i -----
    samtools view -@ $threads -b -o $workdir/data/$i/$i'.bam' -S $workdir/data/$i/$i'.sam' 2> $workdir/data/$i/samtools.out.txt 1> /dev/null
    samtools sort -@ $threads -o $workdir/data/$i/$i'.sorted.bam' $workdir/data/$i/$i'.bam' 2>> $workdir/data/$i/samtools.out.txt 1> /dev/null
    samtools index $workdir/data/$i/$i'.sorted.bam' 2>> $workdir/data/$i/samtools.out.txt 1> /dev/null
    echo 'Done'

    echo ----- REMOVING OBSOLETE READ FILES -----
    rm -rf $workdir/data/$i/$i'.fastq.gz'
    rm -rf $workdir/data/$i/$i'.sam'
    rm -rf $workdir/data/$i/$i'.bam'
    echo 'Done'
done < $sra
echo 'Mapping done'
