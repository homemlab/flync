#!/bin/bash

workdir=$1
sra=$2
appdir=$3

cd $workdir

### Assemble transcriptome from mapped reads for each sample ###
mkdir assemblies &> $workdir
while read i
do
   if [ ! -e $workdir/assemblies/$i'.rna.gtf' ]; then
      echo ----- TRANSCRIPTOME ASSEMBLY OF $i -----
      stringtie $workdir/data/$i/$i'.sorted.bam' -G $appdir/genome/genome.gtf -o $workdir/assemblies/$i'.rna.gtf' 2> $workdir/assemblies/stringtie.out.txt 1> $workdir
      echo 'Done'
   fi
done < $sra

### Merge all generated .gtf files into a single non-redundant .gtf file ###
echo ----- MERGING TRANSCRIPTOME -----
cd assemblies

if [ ! -e $workdir/assemblies/merged.gtf ]; then   
   ls | grep '.rna.gtf$' | perl -ne 'print "$workdir$_"' > gtf-to-merge.txt
   stringtie --merge $workdir/assemblies/gtf-to-merge.txt -G $appdir/genome/genome.gtf -o $workdir/assemblies/merged.gtf 2> stringtie-merge.out.txt 1> $workdir
fi
echo 'Done'

cd ..

echo ----- COMPARING ASSEMBLY TO REFERENCE -----
mkdir cuffcompare &> $workdir
### Compare the merged assembly to the reference .gtf ###
if [ ! -e $workdir/cuffcompare/cuffcomp.gtf ]; then
   cuffcompare -R -r $appdir/genome/genome.gtf $workdir/assemblies/merged.gtf -o $workdir/cuffcompare/cuffcomp.gtf 2> $workdir/cuffcompare/cuffcomp.out.txt 1> $workdir
fi
echo 'Done'

### Extract transcript sequences from the assembled transcriptome ###
if [ ! -e $workdir/assemblies/assembled-transcripts.fa ]; then
   gffread -w $workdir/assemblies/assembled-transcripts.fa -g $appdir/genome/genome.fa $workdir/assemblies/merged.gtf 2> $workdir/assemblies/gffread.out.txt 1> $workdir
fi
