#!/bin/bash

workdir=$1
appdir=$2
threads=$3

cd $workdir

### CPAT analysis (coding probability assessment tool) ###
mkdir -p cpat 
cd cpat
## Get Dmel CPAT files
# logitModel
if [ ! -e Fly_logitModel.RData ]; then
   wget --no-check-certificate -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/Fly_logitModel.RData/download -O Fly_logitModel.RData
fi

# hexamer.tsv
if [ ! -e fly_Hexamer.tsv ]; then
   wget --no-check-certificate -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/fly_Hexamer.tsv/download -O fly_Hexamer.tsv
fi

# fly_cutoff - probability cpat values below this cutoff are considered non-coding
if [ ! -e fly_cutoff.txt ]; then
   wget --no-check-certificate -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/fly_cutoff.txt/download -O fly_cutoff.txt
fi

## Run CPAT - Minimum ORF size = 25; Top ORFs to retain =1
touch $workdir/cpat/cpat.ORF_prob.tsv
echo ----- RUNNING CODING PROBABILITY -----
cpat.py --verbose 0 -x $workdir/cpat/fly_Hexamer.tsv -d $workdir/cpat/Fly_logitModel.RData -g $workdir/assemblies/assembled-new-transcripts.fa -o $workdir/cpat/cpat

echo 'Done'
echo 'Assembly and coding probability done'
