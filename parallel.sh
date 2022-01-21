#!/usr/bin/env bash

## VARIABLES ##
workdir=$1
sra=$(readlink -f $2)
threads=$3
appdir=$4
#metadata=$(readlink -f $5)
#genome=$(readlink -f $6)
#annot=$(readlink -f $7)

## HARDCODED VARS ##
#if [[ -z $SLURM_NTASKS ]]; then
#  jobs=$(expr $(lscpu | grep 'CPU(s):' | awk {'print($2)'}) / 4)
#  jobs=${jobs%.*}
#else
#  if [[ $threads="" ]]; then
#    jobs=$(expr $SLURM_NTASKS / 4)
#    jobs=${jobs%.*}
#  else
#    jobs=$(expr $threads / 4)
#    jobs=${jobs%.*}
#  fi
#fi
#echo 'Number of tasks for this job: ' $jobs

jobs=$(wc -l $sra | cut -f1 -d' ')

downstream_threads=$(expr $threads / $jobs)
downstream_threads=${downstream_threads%.*}

echo 'Number of threads per task: ' $downstream_threads

mkdir -p $appdir/genome

## TESTS ##
# Checking if user supplied genome/annotation files
# if [[ -z $genome ]]; then
#   if ! [[ -z $annot ]]; then
#     echo "To avoid compatibility errors .fa and .gtf files should com from same assembly. Using pre-built genome assembly"
#     bash $appdir/scripts/get-genome.sh $appdir
#     genome=$appdir/genome/genome.fa
#     annot=$appdir/genome/genome.gtf
#   else
#     bash $appdir/scripts/get-genome.sh $appdir
#     genome=$appdir/genome/genome.fa
#     annot=$appdir/genome/genome.gtf
#   fi
# else
#   if [[ -z $annot ]];then
#     echo "To avoid compatibility errors .fa and .gtf files should com from same assembly. Using pre-built genome assembly"
#     bash $appdir/scripts/get-genome.sh $appdir
#     genome=$appdir/genome/genome.fa
#     annot=$appdir/genome/genome.gtf
#   else
#     if ! [[ -f $annot ]]; then
#       echo "-a must be a FILE in the .gtf format"
#       echo "$usage" >&2; exit 1
#     elif ! [[ -f $genome ]]; then
#       echo "-g must be a FILE in the .fa format"
#       echo "$usage" >&2; exit 1
#     else
#       echo "Using user supplied Genome/Annoation assembly"
#       rm -rf $appdir/genome/genome*
#       cp -f $genome $appdir/genome/genome.fa
#       cp -f $annot $appdir/genome/genome.gtf
#     fi
#   fi
# fi

## INITIATE PIPELINE ##
function silence_parallel () {
    ## SILENCE PARALLEL FIRST RUN ##
    parallel --citation &> $appdir/cmd.out
    echo will cite &> $appdir/cmd.out
    rm cmd.out
}

mkdir -p $workdir
conda_path=$(conda info | grep -i 'base environment' | awk '{print$(4)}')
conda_sh=$conda_path'/etc/profile.d/conda.sh'

source $conda_sh
conda init $(echo $SHELL | awk -F'[/]' '{print$(NF)}') &> $appdir/cmd.out

conda activate infoMod
$appdir/scripts/get-genome.sh $appdir
$appdir/scripts/get-sra-info.sh $workdir $sra

conda activate mapMod

## SILENCE PARALLEL FIRST RUN ##
parallel --citation &> cmd.out
echo will cite &> cmd.out
rm cmd.out

$appdir/scripts/build-index.sh $appdir $threads
parallel -k --lb -j $jobs -a $sra $appdir/tux2map.sh $workdir {} $downstream_threads $appdir

conda activate assembleMod

silence_parallel

parallel -k --lb -j $jobs -a $sra $appdir/tux2assemble.sh $workdir {} $downstream_threads $appdir
$appdir/tux2merge.sh $workdir $sra $threads $appdir

conda activate codMod
$appdir/coding-prob.sh $workdir $appdir $threads

$appdir/class-new-transfrags.sh $workdir $threads $appdir

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
