#!/bin/bash

workdir=$1
sra=$(readlink -f $2)
appdir=$3
threads=$4

cd $workdir

### Assemble transcriptome from mapped reads for each sample ###
mkdir assemblies/cufflinks >&2 $workdir/err.log
while read i
do
   if [ ! -e $workdir/assemblies/cufflinks/$i'_cuff'/transcripts.gtf ]; then
      echo ----- TRANSCRIPTOME ASSEMBLY OF $i -----
      readlen=$(cat $workdir/results/runinfo.csv | grep $i | cut -f2 -d,)
      sd=$(bc -l <<< 'scale=2; '$readlen'*0.1*2')
      sd=${sd%.*}
      echo $readlen
      echo $sd
      cufflinks -p $threads -m $readlen -s $sd -o $workdir/assemblies/cufflinks/$i'_cuff' -g $appdir/genome/genome.gtf $workdir/data/$i/$i'.sorted.bam' 2> $workdir/assemblies/cufflinks/cuff.out.txt 1> $workdir
      echo $workdir/assemblies/cufflinks/$i'_cuff'/transcripts.gtf >> $workdir/assemblies/cufflinks/cuffmerge.txt
      echo 'Done'
   fi
done < $sra

### Merge all generated .gtf files into a single non-redundant .gtf file ###
echo ----- MERGING TRANSCRIPTOME -----
cd assemblies

# if [ ! -e $workdir/assemblies/cufflinks/merged.gtf ]; then   
#    ls | grep '.rna.gtf$' | perl -ne 'print "$workdir$_"' > gtf-to-merge.txt
cuffmerge -o $workdir/assemblies/cufflinks/merge -g $appdir/genome/genome.gtf -s $appdir/genome/genome.fa -p $threads $workdir/assemblies/cufflinks/cuffmerge.txt
# fi
echo 'Done'

# cd ..

# echo ----- COMPARING ASSEMBLY TO REFERENCE -----
# mkdir cuffcompare >&2 $workdir/err.log
# ### Compare the merged assembly to the reference .gtf ###
# if [ ! -e $workdir/cuffcompare/cuffcomp.gtf ]; then
#    cuffcompare -R -r $appdir/genome/genome.gtf $workdir/assemblies/cufflinks/merged.gtf -o $workdir/cuffcompare/cuffcomp.gtf 2> $workdir/cuffcompare/cuffcomp.out.txt 1> $workdir
# fi
# echo 'Done'

# ### Extract transcript sequences from the assembled transcriptome ###
# if [ ! -e $workdir/assemblies/cufflinks/assembled-transcripts.fa ]; then
#    gffread -w $workdir/assemblies/cufflinks/assembled-transcripts.fa -g $appdir/genome/genome.fa $workdir/assemblies/cufflinks/merged.gtf 2> $workdir/assemblies/cufflinks/gffread.out.txt 1> $workdir
# fi
