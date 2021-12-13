#|/bin/bash

workdir=$1
sra=$2
appdir=$3
threads=$4

cd $workdir

mkdir kallisto >&2 $workdir/err.log
cd kallisto

### Extract transcript sequences from filtered .gtf file with candidate lincRNAs ###
if [ ! -e $workdir/results/all-new-transcripts.fa ]; then
    gffread -w $workdir/results/all-new-transcripts.fa -g $appdir/genome/genome.fa $workdir/results/all-new-transcripts.gtf
fi

### Run Kallisto index ###
if [ ! -e $workdir/kallisto/all-new-transcripts.fa.idx ]; then
    kallisto index -i $workdir/kallisto/all-new-transcripts.fa.idx $workdir/results/all-new-transcripts.fa
fi

### Run Kallisto to quantify each RNA-seq experiment ###
echo ----- RUNNING PSEUDOALIGNMENT -----
while read i
do
    if [ ! -d $workdir/kallisto/$i ]; then
        lay=$(cat $workdir/results/runinfo.csv | grep $i | cut -f6 -d,)
        readlen=$(cat $workdir/results/runinfo.csv | grep $i | cut -f2 -d,)
        sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
        sd=${sd%.*}
        if [ $lay == 'SINGLE' ]; then
            kallisto quant -b 25 -t $threads -o $workdir/kallisto/$i -i $workdir/kallisto/all-new-transcripts.fa.idx -o $i --single -l $readlen -s $sd $workdir/data/$i/$i'.fastq.gz'
        else
            kallisto quant -b 50 -t $threads -o $workdir/kallisto/$i -i $workdir/kallisto/all-new-transcripts.fa.idx -o $i -l $readlen -s $sd $workdir/data/$i/$i'_1.fastq.gz' $workdir/data/$i/$i'_2.fastq.gz'
        fi
    fi
done < $sra
echo 'Done'
