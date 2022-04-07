#!/usr/bin/env bash

## VARIABLES ##
workdir=$(readlink -f $1)
sra=$(readlink -f $2)
threads=$3
appdir=$(readlink -f $4)
metadata=$(readlink -f $5)
#genome=$(readlink -f $6)
#annot=$(readlink -f $7)

jobs=$(wc -l $sra | cut -f1 -d' ')

echo 'Number of tasks for this job: ' $jobs
if [[ $threads -ge $jobs ]]; then
    downstream_threads=$(expr $threads / $jobs)
    downstream_threads=${downstream_threads%.*}
else
    downstream_threads=1
fi

echo 'Number of threads per task: ' $downstream_threads

mkdir -p $appdir/genome

## INITIATE PIPELINE ##

mkdir -p $workdir
conda_path=$(conda info | grep -i 'base environment' | awk '{print$(4)}')
conda_sh=$conda_path'/etc/profile.d/conda.sh'

source $conda_sh
conda init $(echo $SHELL | awk -F'[/]' '{print$(NF)}') &> $appdir/cmd.out

conda activate infoMod

## SILENCE PARALLEL FIRST RUN ##
parallel --citation &> $appdir/cmd.out
echo will cite &> $appdir/cmd.out
rm $appdir/cmd.out

## RUN SCRIPTS FOR GETTING GENOME AND INFO ON SRA RUNS
$appdir/scripts/get-genome.sh $appdir
$appdir/scripts/get-sra-info.sh $workdir $sra

conda activate mapMod

# ## SILENCE PARALLEL FIRST RUN ##
# parallel --citation &> $appdir/cmd.out
# echo will cite &> $appdir/cmd.out
# rm $appdir/cmd.out

$appdir/scripts/build-index.sh $appdir $threads
parallel -k --lb -j $jobs -a $sra $appdir/tux2map.sh $workdir {} $downstream_threads $appdir

conda activate assembleMod

parallel -k --lb -j $jobs -a $sra $appdir/tux2assemble.sh $workdir {} $downstream_threads $appdir
$appdir/tux2merge.sh $workdir $sra $threads $appdir

conda activate codMod

$appdir/coding-prob.sh $workdir $appdir $threads

$appdir/class-new-transfrags.sh $workdir $threads $appdir

conda activate assembleMod

### Extract transcript sequences from filtered .gtf file with new non-coding & coding transcripts ###
if [ ! -e $workdir/results/new-non-coding-transcripts.fa ]; then
    gffread -w $workdir/results/new-non-coding-transcripts.fa -g $appdir/genome/genome.fa $workdir/results/new-non-coding.gtf
fi

if [ ! -e $workdir/results/new-coding-transcripts.fa ]; then
    gffread -w $workdir/results/new-coding-transcripts.fa -g $appdir/genome/genome.fa $workdir/results/new-coding.gtf
fi

conda activate dgeMod

if [[ -z {$metadata+x} ]]; then
    echo "Skipping DGE since no metadata file was provided..."
else
    $appdir/dea.sh $workdir $sra $appdir $threads $metadata
fi

conda deactivate
#
#$appdir/dea.sh $workdir $sra $appdir $threads $metadata
# 
# ls $workdir/results/*.gtf* > $appdir/gtfs.txt
# parallel -k --lb -j $jobs -a $appdir/gtfs.txt $appdir/scripts/gtf-to-bed.sh {}
# rm $appdir/gtfs.txt
# 
# parallel -k --lb -j $jobs -a $appdir/tracksFile.tsv $appdir/get-features.sh {} $workdir/results/new-non-coding.chr.bed $workdir
# 
# parallel -k --lb -j $jobs -a $appdir/tracksFile.tsv $appdir/get-features.sh {} $workdir/results/new-coding.chr.bed $workdir
