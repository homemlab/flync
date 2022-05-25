#!/bin/bash

## Prepare BED files for getting feature tables for ML-training and testing

input=$(readlink -f $1)
outfile=$(echo ${input%.*} | awk -F'[/]' '{print$(NF)}')

# Use BEDOPS convert2bed script to convert gtf to bed
# Needed to add the score integer isntead of '.'

cat $input | convert2bed -i gtf | awk '$5="1000"' | sed 's/\ gene_name\ [^~]*"//g' | awk '$7=$13$15' | cut -f1-7 -d' ' | sed 's/;$//g' | sed 's/;$//g' | sed 's/;/./g' | sed 's/"//g' | awk '$4=$7' | cut -f1-6 -d' ' | sed 's/\ /\t/g' | sort-bed - > $outfile'.bed'
cat $input | convert2bed -i gtf | awk '$5="1000" && $1="chr"$1' | sed 's/\ gene_name\ [^~]*"//g' | awk '$7=$13$15' | cut -f1-7 -d' ' | sed 's/;$//g' | sed 's/;$//g' | sed 's/;/./g' | sed 's/"//g' | awk '$4=$7' | cut -f1-6 -d' ' | sed 's/\ /\t/g' | sort-bed - > $outfile'.chr.bed'

# Used this to output unique names using <transcript_id>.<exon_number>
# Otherwise, USCS binaries (i.e. bigWigAverageOverBed) would not work properly
