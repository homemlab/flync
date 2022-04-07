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
echo ----- RUNNING CODING PROBABILITY -----
if [ ! -e cpat.ORF_prob.best.tsv ]; then
   cpat.py --verbose 0 -x $workdir/cpat/fly_Hexamer.tsv -d $workdir/cpat/Fly_logitModel.RData -g $workdir/assemblies/assembled-transcripts.fa -o $workdir/cpat/cpat --min-orf 25 --top-orf 1
fi
echo 'Done'
echo 'Assembly and coding probability done'

## Including FEELnc
# mkdir -p $workdir/FEELnc
# FEELnc_filter.pl -i $workdir/assemblies/merged.gtf -a $appdir/genome/genome.gtf -b transcript_biotype=protein_coding --monoex=1 -p $threads > $workdir/FEELnc/FEELnc_candidate_lncRNA.gtf
# FEELnc_codpot.pl -i $workdir/FEELnc/FEELnc_candidate_lncRNA.gtf -g $appdir/genome/genome.fa -a $appdir/genome/genome.cds.gtf -l $appdir/genome/genome.lncrna.gtf -n 1500,1500 --outdir=$workdir/FEELnc
# FEELnc_classifier.pl -i $workdir/FEELnc/FEELnc_candidate_lncRNA.gtf.lncRNA.gtf -a $appdir/genome/genome.gtf > FEElnc_lncRNA_class.txt
# FEELnc_classifier.pl -i $workdir/FEELnc/FEELnc_candidate_lncRNA.gtf.noORF.gtf -a $appdir/genome/genome.gtf >> FEElnc_lncRNA_class.txt
