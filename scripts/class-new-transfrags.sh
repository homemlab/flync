#!/usr/bin/env bash

workdir=$1

cd $workdir

### Save .gtf files for each type of new transcript discovered
grep '; class_code "u";' $workdir/cuffcompare/cuffcomp.gtf.combined.gtf > results/candidate-lincRNA.gtf
grep '; class_code "i";' $workdir/cuffcompare/cuffcomp.gtf.combined.gtf > results/candidate-intronicRNA
grep '; class_code "x";' $workdir/cuffcompare/cuffcomp.gtf.combined.gtf > results/candidate-antisenseRNA.gtf
grep '; class_code "j";' $workdir/cuffcompare/cuffcomp.gtf.combined.gtf > results/candidate-isoforms.gtf

### Combine CPAT reported NO_ORF transcripts with low probability '('< cutoff')'
grep 'MSTRG*' cpat/cpat.no_ORF.txt > cpat/cpat.no_ORF.final.txt
awk '$11 <= 0.39 {print ($1)}' cpat/cpat.ORF_prob.best.tsv >> cpat/cpat.no_ORF.final.txt 
sort -g cpat/cpat.no_ORF.final.txt | sort -t . -k 2 -g | uniq | grep 'MSTRG*'> cpat/cpat.non-coding.sorted.txt
rm -f cpat/cpat.no_ORF.final.txt
awk '$11 > 0.39 {print ($1)}' cpat/cpat.ORF_prob.best.tsv > cpat/cpat.ORF.final.txt 
sort -g cpat/cpat.ORF.final.txt | sort -t . -k 2 -g | uniq | grep 'MSTRG*'> cpat/cpat.coding.sorted.txt
rm -f cpat/cpat.ORF.final.txt
awk '$11 > 0.39 && $8 <= 150 {print ($1)}' cpat/cpat.ORF_prob.best.tsv > cpat/cpat.uORF.final.txt 
sort -g cpat/cpat.uORF.final.txt | sort -t . -k 2 -g | uniq | grep 'MSTRG*'> cpat/cpat.coding.microORFs.sorted.txt
rm -f cpat/cpat.uORF.final.txt

mkdir -p $workdir/results/{coding,non-coding}

grep -f $workdir/cpat/cpat.non-coding.sorted.txt $workdir/results/candidate-lincRNA.gtf > $workdir/results/non-coding/new-lincRNAs.gtf
grep -f $workdir/cpat/cpat.non-coding.sorted.txt $workdir/results/candidate-intronicRNA > $workdir/results/non-coding/new-intronicRNAs.gtf
grep -f $workdir/cpat/cpat.non-coding.sorted.txt $workdir/results/candidate-antisenseRNA.gtf > $workdir/results/non-coding/new-antisenseRNAs.gtf
grep -f $workdir/cpat/cpat.coding.sorted.txt $workdir/results/candidate-isoforms.gtf > $workdir/results/coding/new-isoforms.gtf
grep -f $workdir/cpat/cpat.coding.microORFs.sorted.txt $workdir/results/candidate-lincRNA.gtf > $workdir/results/coding/new-lncRNA-microORFs.gtf
grep -f $workdir/cpat/cpat.coding.microORFs.sorted.txt $workdir/results/candidate-intronicRNA >> $workdir/results/coding/new-lncRNA-microORFs.gtf
grep -f $workdir/cpat/cpat.coding.microORFs.sorted.txt $workdir/results/candidate-antisenseRNA.gtf >> $workdir/results/coding/new-lncRNA-microORFs.gtf

rm $workdir/results/candidate*

cd $workdir/results/non-coding/
cat new-lincRNAs.gtf new-intronicRNAs.gtf new-antisenseRNAs.gtf | sort -t . -k 2 -g | uniq > $workdir/results/new-non-coding.gtf

cd $workdir/results/coding/
cat new-isoforms.gtf new-lncRNA-microORFs.gtf | sort -t . -k 2 -g | uniq > $workdir/results/new-coding.gtf
