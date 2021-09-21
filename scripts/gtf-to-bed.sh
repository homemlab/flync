#!/bin/bash

## Prepare BED files for getting feature tables for ML-training and testing

outfile=`echo $1|sed 's/\(.*\)\..*/\1/'`
input=$(readlink -f $1)

# Use BEDOPS convert2bed script to convert gtf to bed
# Needed to add the score integer isntead of '.'

cat $input | convert2bed -i gtf > $outfile'.2.bed'
awk '$1="chr"$1' $outfile'.2.bed' > $outfile'.2.chr.bed'
awk '$5="1000"' $outfile'.2.bed' > $outfile'.3.bed'
awk '$5="1000"' $outfile'.2.chr.bed' > $outfile'.3.chr.bed'
rm $outfile'.2.bed' $outfile'.2.chr.bed'

# Used this to output unique names using <transcript_id>.<exon_number>
# Otherwise, USCS binaries (i.e. bigWigAverageOverBed) would not work properly

cat $outfile'.3.bed' | sed 's/\ gene_name\ [^~]*"//g' | awk '$7=$13$15' | cut -f1-7 -d' ' | sed 's/;;//g' | sed 's/;/./g' | sed 's/"//g' | awk '$4=$7' | cut -f1-6 -d' ' | sed 's/\ /\t/g'| sort-bed - > $outfile'.bed'
cat $outfile'.3.chr.bed' | sed 's/\ gene_name\ [^~]*"//g' | awk '$7=$13$15' | cut -f1-7 -d' ' | sed 's/;;//g' | sed 's/;/./g' | sed 's/"//g' | awk '$4=$7' | cut -f1-6 -d' ' | sed 's/\ /\t/g' | sort-bed - > $outfile'.chr.bed'
rm $outfile'.3.bed' $outfile'.3.chr.bed'