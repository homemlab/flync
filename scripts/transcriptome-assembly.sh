#!/bin/bash

workdir=$1
sra=$2
appdir=$3

cd $workdir

### Assemble transcriptome from mapped reads for each sample ###
mkdir assemblies
while read i
do
   echo ----- TRANSCRIPTOME ASSEMBLY OF $i -----
   stringtie $workdir/data/$i/$i'.sorted.bam' -G $appdir/genome/dm6.gtf -o $workdir/assemblies/$i'.rna.gtf'
done < $sra

### Merge all generated .gtf files into a single non-redundant .gtf file ###
echo ----- MERGING TRANSCRIPTOME -----
cd assemblies

### POSSIBLE ERROR ON $workdir VAR
ls | grep '.rna.gtf$' | perl -ne 'print "$workdir$_"' > gtf-to-merge.txt
stringtie --merge $workdir/assemblies/gtf-to-merge.txt -G $appdir/genome/dm6.gtf -o $workdir/assemblies/merged.gtf

## Clean-up
# rm -rf gtf-to-merge.txt
# rm -rf *.rna.gtf

cd ..

echo ----- COMPARING ASSEMBLY TO REFERENCE -----
mkdir cuffcompare
### Compare the merged assembly to the reference .gtf ###
cuffcompare -R -r $appdir/genome/dm6.gtf $workdir/assemblies/merged.gtf -o $workdir/cuffcompare/cuffcomp.gtf

### Extract transcript sequences from the assembled transcriptome ###
gffread -w $workdir/assemblies/assembled-transcripts.fa -g $appdir/genome/dm6.fa $workdir/assemblies/merged.gtf

### CPAT analysis (coding probability assessment tool) ###
mkdir cpat
cd cpat
## Get Dmel CPAT files
# logitModel
wget -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/Fly_logitModel.RData/download -O Fly_logitModel.RData
# hexamer.tsv
wget -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/fly_Hexamer.tsv/download -O fly_Hexamer.tsv
# fly_cutoff - probability cpat values below this cutoff are considered non-coding
wget -q https://sourceforge.net/projects/rna-cpat/files/v1.2.2/prebuilt_model/fly_cutoff.txt/download -O fly_cutoff.txt

## Run CPAT - Minimum ORF size = 25; Top ORFs to retain =1 
cpat.py --verbose false -x $workdir/cpat/fly_Hexamer.tsv -d $workdir/cpat/Fly_logitModel.RData -g $workdir/assemblies/assembled-transcripts.fa -o $workdir/cpat/cpat --min-orf 25 --top-orf 1
