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

### Download and map reads from SRA accession numbers in .txt file ###
mkdir -p data 
mkdir -p data/$2 
echo ----- DOWNLOADING READS $2 -----
if [ ! -e data/$2/*.fastq.gz ] || [ ! -e data/$2/*.fastq ]; then
    ## Faster:
    fasterq-dump -f -3 -p -e $threads -O $workdir/data/$2 $2
    if ! [ -e data/$2/*.fastq.gz ]; then
        gzip $workdir/data/$2/*.fastq
    fi
fi    
echo 'Done'
wait
echo ----- MAPPING READS $2 -----
if [ ! -e data/$2/*.sorted.bam ] && [ ! -e data/$2/*.bam ] && [ ! -e data/$2/*.sam ]; then
    lay=$(grep $2 $workdir/results/runinfo.csv | cut -f6 -d,)
    if [[ $lay == 'SINGLE' ]]; then
        hisat2 -p $threads -x $appdir/genome/genome.idx -U $workdir/data/$2/$2'.fastq.gz' -S $workdir/data/$2/$2'.sam' --dta --known-splicesite-infile $appdir/genome/genome.sj &> $workdir/data/$2.hisat2.log
    else
        hisat2 -p $threads -x $appdir/genome/genome.idx -1 $workdir/data/$2/$2'_1.fastq.gz' -2 $workdir/data/$2/$2'_2.fastq.gz' -S $workdir/data/$2/$2'.sam' --dta --known-splicesite-infile $appdir/genome/genome.sj &> $workdir/data/$2.hisat2.log
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
#rm -rf $workdir/data/$2/$2'.fastq.gz'
rm -rf $workdir/data/$2/$2'.sam'
rm -rf $workdir/data/$2/$2'.bam'
echo 'Done'

mkdir -p assemblies/stringtie
if [ ! -e $workdir/assemblies/stringtie/$2'.rna.gtf' ]; then
    echo ----- STRINGTIE TRANSCRIPTOME ASSEMBLY OF $2 -----
    stringtie $workdir/data/$2/$2'.sorted.bam' -G $appdir/genome/genome.gtf -o $workdir/assemblies/stringtie/$2'.rna.gtf'
    echo 'Done'
fi
wait
if [ ! -e $workdir/assemblies/cufflinks/$2'_cuff'/transcripts.gtf ]; then
    echo ----- CUFFLINKS TRANSCRIPTOME ASSEMBLY OF $2 -----
    cufflinks --no-update-check -q -p $threads -m $readlen -s $sd -o $workdir/assemblies/cufflinks/$2'_cuff' -g $appdir/genome/genome.gtf $workdir/data/$2/$2'.sorted.bam'
    echo $workdir/assemblies/cufflinks/$2'_cuff'/transcripts.gtf >> $workdir/assemblies/cufflinks/cuffmerge.txt
    echo 'Done'
fi