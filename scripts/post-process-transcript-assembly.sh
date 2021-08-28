#!/bin/bash

### Seen it in stackexchange.com '('https://unix.stackexchange.com/questions/88550/vlookup-function-in-unix')' and modified by RFdS
## Explained:
# This will read the first file in the command $awk -f vlookup.awk <file1> <file2> into an array - a - and then will match the 5th column of file 2. If true, will print the line to STDOUT

workdir=$1
progfile=$2
cd $workdir
cutoff=$(grep 'Cutoff' cpat/fly_cutoff.txt | sed 's@^[^0-9]*\([0-9]\+\)*@\1@')
echo cpat cutoff is: $cutoff '('D. melanogaster - see CPAT docs')'

### Save the 'u'-tagged assembled transcripts '('candidate novel lincRNAs')' supported by read coverage
grep -w "u" $workdir/assemblies/cuffcomp.gtf.merged.gtf.tmap | sort -n -k 10 > results/candidate-assembled-lincRNAs.tsv

### Combine CPAT reported NO_ORF transcripts with low probability '('< cutoff')'
if [ -f 'cpat/cpat.ORF_prob.best.tsv' ]
then
awk '$11 <= 0.39 {print ($1)}' cpat/cpat.ORF_prob.best.tsv >> cpat/cpat.no_ORF.final.txt 
sort -g cpat/cpat.no_ORF.final.txt | sort -t . -k 2 -g | uniq > cpat/cpat.non-coding.sorted.txt
fi

sort -t . -k 2 -n results/candidate-assembled-lincRNAs.tsv > results/candidate-assembled-lincRNAs.sorted.tsv

### This latter file is a list of non-coding transcripts that can now be used as reference to search new lncRNAs
awk -f $progfile cpat/cpat.non-coding.sorted.txt results/candidate-assembled-lincRNAs.sorted.tsv > results/new-lincRNAs.tsv

### These lines will extract extract the stringtie ids from the new-lincRNAs.tsv info and use it as a pattern to filter the merged.gtf to a final lincRNA.gtf
awk '{print $5}' results/new-lincRNAs.tsv > results/new-lincRNA-ids.txt
grep -f results/new-lincRNA-ids.txt assemblies/merged.gtf > results/new-lincRNAs.gtf

### Get isolated loci with transcript candidates and filter with new lincRNA ids
awk '$3~/-/' cuffcompare/cuffcomp.gtf.loci | grep -f results/new-lincRNA-ids.txt > results/new-lincRNA-isolated.loci
awk '$3~/-/' cuffcompare/cuffcomp.gtf.loci | grep -f results/new-lincRNA-ids.txt | awk '{print $4}' | awk -F '[,]' '{print $(NF-1)}' > results/new-lincRNA-isolated-ids.txt
grep -f results/new-lincRNA-isolated-ids.txt assemblies/merged.gtf > results/new-lincRNAs-isolated-loci.gtf

### Stats 
a=$(cat results/new-lincRNAs.tsv | wc -l)
b=$(cat results/candidate-assembled-lincRNAs.tsv | wc -l)
c=$(cat cuffcompare/cuffcomp.gtf | grep 'Novel loci:' | sed 's@^[^0-9]*\([0-9]\+\).*@\1@')

echo -e $c novel loci have been identified,'\n''\t'of which $b novel transcripts where found 'in' intergenic regions,'\n''\t'and $a were non-coding '('assessed by CPAT:3.0.4')''\n'A total of $a lincRNAs have been discovered'!' > results/results-summary.log