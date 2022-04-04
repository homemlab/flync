#!/bin/bash

workdir=$1
sra=$2
threads=$3
appdir=$4

cd $workdir
mkdir -p assemblies/stringtie
echo ----- MERGING TRANSCRIPTOME ASSEMBLIES -----

cd $workdir/assemblies/stringtie

ls | grep '.rna.gtf$' | perl -ne 'print "$workdir$_"' > gtf-to-merge.txt

if [ ! -e $workdir/assemblies/stringtie/merged.gtf ]; then   
   stringtie --merge $workdir/assemblies/stringtie/gtf-to-merge.txt -G $appdir/genome/genome.gtf -o $workdir/assemblies/stringtie/merged.gtf
fi

cd $workdir/assemblies/cufflinks
mkdir -p $workdir/assemblies/cufflinks/merged

if [ ! -d $workdir/assemblies/cufflinks/merged.gtf ]; then
   stringtie --merge $workdir/assemblies/cufflinks/cuffmerge.txt -G $appdir/genome/genome.gtf -o $workdir/assemblies/cufflinks/merged.gtf
fi
echo 'Done'
cd $workdir/assemblies

echo ----- MERGING STRINGTIE WITH CUFFLINKS TRANSFRAGS -----
echo $workdir/assemblies/stringtie/merged.gtf >> $workdir/assemblies/final_merge.txt
echo $workdir/assemblies/cufflinks/merged.gtf >> $workdir/assemblies/final_merge.txt
if [ ! -e $workdir/assemblies/merged.gtf ]; then
   stringtie --merge $workdir/assemblies/final_merge.txt -G $appdir/genome/genome.gtf -o $workdir/assemblies/merged.gtf
fi
echo 'Done'

echo ----- COMPARING ASSEMBLY TO REFERENCE -----
mkdir -p $workdir/cuffcompare 
### Compare the merged assembly to the reference .gtf ###
if [ ! -e $workdir/cuffcompare/cuffcomp.gtf ]; then
   cuffcompare -R -r $appdir/genome/genome.gtf $workdir/assemblies/merged.gtf -o $workdir/cuffcompare/cuffcomp.gtf
fi
echo 'Done'

### Extract transcript sequences from the assembled transcriptome ###
if [ ! -e $workdir/assemblies/assembled-transcripts.fa ]; then
   gffread -w $workdir/assemblies/assembled-transcripts.fa -g $appdir/genome/genome.fa $workdir/assemblies/merged.gtf
fi



