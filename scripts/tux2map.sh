#!/usr/bin/env bash

workdir=$1
sra=$2
threads=$3
appdir=$4
samthr=$(expr $threads / 2)
if [[ $samthr < 1 ]]; then
    samthr=1
fi
readlen=$(grep $2 $workdir/results/runinfo.csv | cut -f2 -d,)
sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
sd=${sd%.*}


cd $workdir

### Download and map reads from SRA accession numbers in .txt file ###
mkdir -p data/$2 
echo ----- DOWNLOADING READS $2 -----
if [[ ! -e data/$2/*.fastq.gz ]] || [[ ! -e data/$2/*1.fastq.gz ]] || [[ ! -e data/$2/*2.fastq.gz ]]; then
    ## Faster:
    cd $workdir
    prefetch -O $workdir/data/$2 $2 &>> $workdir/run.log
    wait
    cd $workdir/data/$2
    fasterq-dump -f -3 -e $threads -O $workdir/data/$2 $2 &>> $workdir/run.log
    if ! [ -e data/$2/*.fastq.gz ]; then
       gzip $workdir/data/$2/*.fastq 
    fi
	## Even faster:
	#parallel-fastq-dump --tmpdir . --threads $threads --gzip --split-files --sra-id $2 --outdir $workdir/data/$2
fi    
echo 'Done'
cd $workdir
wait
echo ----- MAPPING READS $2 -----
if [ ! -e data/$2/*.sorted.bam ] || [ ! -e data/$2/*.bam ] || [ ! -e data/$2/*.sam ]; then
    lay=$(grep $2 $workdir/results/runinfo.csv | cut -f6 -d,)
    if [[ $lay == 'SINGLE' ]]; then
        hisat2 -p $threads -x $appdir/genome/genome.idx -U $workdir/data/$2/$2'.fastq.gz' -S $workdir/data/$2/$2'.sam' --dta --dta-cufflinks --known-splicesite-infile $appdir/genome/genome.ss &> $workdir/data/$2.hisat2.log
    else
        hisat2 -p $threads -x $appdir/genome/genome.idx -1 $workdir/data/$2/$2'_1.fastq.gz' -2 $workdir/data/$2/$2'_2.fastq.gz' -S $workdir/data/$2/$2'.sam' --dta --dta-cufflinks --known-splicesite-infile $appdir/genome/genome.ss &> $workdir/data/$2.hisat2.log
    fi
fi
echo 'Done'
wait
echo ----- CONVERTING .SAM TO INDEXED SORTED .BAM OF $2 -----
if [ ! -e data/$2/*.sorted.bam ] || [ ! -e data/$2/*.bam ]; then
    samtools view -@ $samthr -b -o $workdir/data/$2/$2'.bam' -S $workdir/data/$2/$2'.sam' 2> $workdir/data/$2/samtools.out.txt 1> /dev/null
    samtools sort -@ $samthr -o $workdir/data/$2/$2'.sorted.bam' $workdir/data/$2/$2'.bam' 2>> $workdir/data/$2/samtools.out.txt 1> /dev/null
    samtools index $workdir/data/$2/$2'.sorted.bam' 2>> $workdir/data/$2/samtools.out.txt 1> /dev/null
fi
echo 'Done'
wait
echo ----- REMOVING OBSOLETE READ FILES -----
rm -rf $workdir/data/$2/$2'.fastq.gz'
rm -rf $workdir/data/$2/$2'.sam'
rm -rf $workdir/data/$2/$2'.bam'
echo 'Done'
