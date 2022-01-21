#!/bin/bash

workdir=$1
sra=$2
appdir=$3
threads=$4
metadata=$5

cd $workdir

mkdir -p kallisto/{non-coding,coding}
cd kallisto

### Extract transcript sequences from filtered .gtf file with new non-coding & coding transcripts ###
if [ ! -e $workdir/results/new-non-coding-transcripts.fa ]; then
    gffread -w $workdir/results/new-non-coding-transcripts.fa -g $appdir/genome/genome.fa $workdir/results/new-non-coding.gtf
fi

if [ ! -e $workdir/results/new-coding-transcripts.fa ]; then
    gffread -w $workdir/results/new-coding-transcripts.fa -g $appdir/genome/genome.fa $workdir/results/new-coding.gtf
fi

### Run Kallisto index on non-coding & coding ###
if [ ! -e $workdir/kallisto/non-coding/new-non-coding-transcripts.fa.idx ]; then
    kallisto index -i $workdir/kallisto/non-coding/new-non-coding-transcripts.fa.idx $workdir/results/new-non-coding-transcripts.fa
fi

if [ ! -e $workdir/kallisto/coding/new-coding-transcripts.fa.idx ]; then
    kallisto index -i $workdir/kallisto/coding/new-coding-transcripts.fa.idx $workdir/results/new-coding-transcripts.fa
fi

### Run Kallisto to quantify each RNA-seq experiment for non-coding transcripts ###
echo ----- RUNNING PSEUDOALIGNMENT ON NEW NON-CODING TRANSCRIPTS -----
while read i
do
    if [ ! -d $workdir/kallisto/non-coding/$i ]; then
        lay=$(cat $workdir/results/runinfo.csv | grep $i | cut -f6 -d,)
        readlen=$(cat $workdir/results/runinfo.csv | grep $i | cut -f2 -d,)
        sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
        sd=${sd%.*}
        if [ $lay == 'SINGLE' ]; then
            kallisto quant -b 25 -t $threads -o $workdir/kallisto/non-coding/$i -i $workdir/kallisto/non-coding/new-non-coding-transcripts.fa.idx --single -l $readlen -s $sd $workdir/data/$i/$i'.fastq.gz'
        else
            kallisto quant -b 50 -t $threads -o $workdir/kallisto/non-coding/$i -i $workdir/kallisto/non-coding/new-non-coding-transcripts.fa.idx -l $readlen -s $sd $workdir/data/$i/$i'_1.fastq.gz' $workdir/data/$i/$i'_2.fastq.gz'
        fi
    fi
done < $sra
echo 'Done'

echo ----- RUNNING PSEUDOALIGNMENT ON NEW CODING TRANSCRIPTS -----
while read i
do
    if [ ! -d $workdir/kallisto/coding/$i ]; then
        lay=$(cat $workdir/results/runinfo.csv | grep $i | cut -f6 -d,)
        readlen=$(cat $workdir/results/runinfo.csv | grep $i | cut -f2 -d,)
        sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
        sd=${sd%.*}
        if [ $lay == 'SINGLE' ]; then
            kallisto quant -b 25 -t $threads -o $workdir/kallisto/coding/$i -i $workdir/kallisto/coding/new-coding-transcripts.fa.idx --single -l $readlen -s $sd $workdir/data/$i/$i'.fastq.gz'
        else
            kallisto quant -b 50 -t $threads -o $workdir/kallisto/coding/$i -i $workdir/kallisto/coding/new-coding-transcripts.fa.idx -l $readlen -s $sd $workdir/data/$i/$i'_1.fastq.gz' $workdir/data/$i/$i'_2.fastq.gz'
        fi
    fi
done < $sra
echo 'Done'


### Pre-processing for Sleuth analysis ###
cd $workdir
mkdir -p results_dea 
cd results_dea

### Create a .csv file for sleuth analysis of nnew on-coding and coding transcripts ###
echo sample,condition,path > $workdir/results_dea/metadata-non-coding.csv
while read i
do
    srr=$(grep $i $metadata | cut -f1 -d,)
    srrcond=$(grep $i $metadata | cut -f2 -d,)
    
    if [ $( echo $srr | wc -l ) == 1 ]; then
        echo $srr','$srrcond','$workdir/kallisto/non-coding/$srr >> $workdir/results_dea/metadata-non-coding.csv
    fi
done < $sra

echo sample,condition,path > $workdir/results_dea/metadata-coding.csv
while read i
do
    srr=$(grep $i $metadata | cut -f1 -d,)
    srrcond=$(grep $i $metadata | cut -f2 -d,)
    
    if [ $( echo $srr | wc -l ) == 1 ]; then
        echo $srr','$srrcond','$workdir/kallisto/coding/$srr >> $workdir/results_dea/metadata-coding.csv
    fi
done < $sra

wait
Rscript $appdir/scripts/sleuth.r $workdir $workdir/results_dea/metadata-non-coding.csv
wait
mv $workdir/results_dea/dea_all.csv $workdir/results_dea/non-coding_dea_all.csv
mv $workdir/results_dea/dea_sig_0.05.csv $workdir/results_dea/non-coding_dea_sig_0.05.csv
wait
Rscript $appdir/scripts/sleuth.r $workdir $workdir/results_dea/metadata-coding.csv
wait
mv $workdir/results_dea/dea_all.csv $workdir/results_dea/coding_dea_all.csv
mv $workdir/results_dea/dea_sig_0.05.csv $workdir/results_dea/coding_dea_sig_0.05.csv
