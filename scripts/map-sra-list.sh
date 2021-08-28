#!/bin/bash

workdir=$1
sra=$2
threads=$3
appdir=$4
samthr=$(expr $threads / 2)

cd $workdir

### Download and map reads from SRA accession numbers in .txt file ###

mkdir data &> /dev/null
while read i
do
    mkdir data/$i &> /dev/null
    echo ----- DOWNLOADING READS $i -----
    if ! [ -f $i'.fastq' ]; then
        ## Faster:
        fasterq-dump -f -3 -p -e $threads -O $workdir/data/$i $i
        if ! [ -f $i'.fastq.gz' ]; then
            gzip $workdir/data/$i/*.fastq
        fi
    fi    
    ## Old command:
    #fastq-dump --gzip -O $workdir/data/$i $i 2> $workdir/data/$i/fastq.out.txt 1> /dev/null
    echo 'Done'

    echo ----- QC REPORT OF $i -----
    fastqc -t $threads $workdir/data/$i/*.fastq.gz 2> $workdir/data/$i/fastq.out.txt 1> /dev/null
    echo 'Done'    
    
    echo ----- MAPPING READS $i -----
    lay=$(cat $workdir/results/runinfo.csv | grep $i | cut -f6 -d,)
    if [ $lay == 'SINGLE' ]; then
        hisat2 -p $threads -x $appdir/genome/genome.idx -U $workdir/data/$i/$i'.fastq.gz' -S $workdir/data/$i/$i'.sam' --dta --known-splicesite-infile $appdir/genome/genome.sj &> $workdir/data/$i.hisat2.log
    else
        hisat2 -p $threads -x $appdir/genome/genome.idx -1 $workdir/data/$i/$i'_1.fastq.gz' -2 $workdir/data/$i/$i'_2.fastq.gz' -S $workdir/data/$i/$i'.sam' --dta --known-splicesite-infile $appdir/genome/genome.sj &> $workdir/data/$i.hisat2.log
    fi
    echo 'Done'

    echo ----- CONVERTING .SAM TO INDEXED SORTED .BAM OF $i -----
    samtools view -@ $samthr -b -o $workdir/data/$i/$i'.bam' -S $workdir/data/$i/$i'.sam' 2> $workdir/data/$i/samtools.out.txt 1> /dev/null
    samtools sort -@ $samthr -o $workdir/data/$i/$i'.sorted.bam' $workdir/data/$i/$i'.bam' 2>> $workdir/data/$i/samtools.out.txt 1> /dev/null
    samtools index $workdir/data/$i/$i'.sorted.bam' 2>> $workdir/data/$i/samtools.out.txt 1> /dev/null
    echo 'Done'

    echo ----- REMOVING OBSOLETE READ FILES -----
    #rm -rf $workdir/data/$i/$i'.fastq.gz'
    rm -rf $workdir/data/$i/$i'.sam'
    rm -rf $workdir/data/$i/$i'.bam'
    echo 'Done'
done < $sra
echo 'Mapping done'
