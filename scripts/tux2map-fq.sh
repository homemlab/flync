#!/bin/bash

workdir=$(readlink -f $1)
fq=$2
threads=$3
appdir=$4
lay=${5^^}
samthr=$(expr $threads / 2)

### FILE NAMES AND EXTENSIONS NOT WORKING... 

fullfile=$(basename -- "$fq")
filename="${fullfile%.*}"
filetype=".""${fullfile##*.}"
filesuff=$(echo $filename | awk -F'[_]' '{print$1}')
filepath=$(echo $fq | awk 'BEGIN{FS=OFS="/"}{$NF=""; NF--; print}')

echo $filename, $filesuff, $filetype, $filepath

cd $workdir
mkdir -p $workdir/data/$filename

echo ----- MAPPING READS $filename -----
if [ ! -e data/$filename/*.sorted.bam ] && [ ! -e data/$filename/*.bam ] && [ ! -e data/$filename/*.sam ]; then
    if [[ $lay == 'SINGLE' ]]; then
        if [ $filetype != *.gz ]; then
            gzip "$filepath""/""$filename"
            hisat2 -p $threads -x $appdir/genome/genome.idx -U $fq -S $workdir/data/$filename/$filename'.sam' --dta --dta-cufflinks --known-splicesite-infile $appdir/genome/genome.ss &> $workdir/data/$filename.hisat2.log
        fi
    else
        if [[ $filename == *"_1" ]]; then
            if [ $filetype != *.gz ]; then
                gzip "$filepath""/""$filesuff""_1""$filetype"
                gzip "$filepath""/""$filesuff""_2""$filetype"
                hisat2 -p $threads -x $appdir/genome/genome.idx -1 "$filepath""/""$filesuff""_1""$filetype"".gz" -2 "$filepath""/""$filesuff""_2""$filetype"".gz" -S $workdir/data/$filename/$filename'.sam' --dta --dta-cufflinks --known-splicesite-infile $appdir/genome/genome.ss &> $workdir/data/$filename.hisat2.log
            fi
            hisat2 -p $threads -x $appdir/genome/genome.idx -1 "$filepath""/""$filesuff""_1""$filetype" -2 "$filepath""/""$filesuff""_2""$filetype" -S $workdir/data/$filename/$filename'.sam' --dta --dta-cufflinks --known-splicesite-infile $appdir/genome/genome.ss &> $workdir/data/$filename.hisat2.log
        fi
    fi
fi
echo 'Done'
wait
echo ----- CONVERTING .SAM TO INDEXED SORTED .BAM OF $filename -----
if [ ! -e data/$filename/*.sorted.bam ] || [ ! -e data/$filename/*.bam ]; then
    samtools view -@ $samthr -b -o $workdir/data/$filename/$filename'.bam' -S $workdir/data/$filename/$filename'.sam' 2> $workdir/data/$filename/samtools.out.txt 1> /dev/null
    samtools sort -@ $samthr -o $workdir/data/$filename/$filename'.sorted.bam' $workdir/data/$filename/$filename'.bam' 2>> $workdir/data/$filename/samtools.out.txt 1> /dev/null
    samtools index $workdir/data/$filename/$filename'.sorted.bam' 2>> $workdir/data/$filename/samtools.out.txt 1> /dev/null
fi
echo 'Done'
wait
echo ----- REMOVING OBSOLETE READ FILES -----
rm -rf $workdir/data/$filename/$filename'.fastq.gz'
rm -rf $workdir/data/$filename/$filename'.sam'
rm -rf $workdir/data/$filename/$filename'.bam'
echo 'Done'