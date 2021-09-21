#!/bin/bash
outfile=`echo $1|sed 's/\(.*\)\..*/\1/'`
bed=$(readlink -f $1)

# Download MAF alignments for each chromosome (USCS does'n have whole genome MAF files for dm6)
## chr2L
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chr2L.maf.gz
gzip -d chr2L.maf.gz

# Build MAF file to feed RNAz (dependent on PHAST pacakge's maf_parse program) getting relevant coordinates from BED file
maf_parse -g $bed chr2L.maf > $outfile'.maf'
rm chr2L.maf

# Repeat for other chromosomes
## chr2R
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chr2R.maf.gz
gzip -d chr2R.maf.gz
maf_parse -g $bed chr2R.maf >> $outfile'.maf'
rm chr2R.maf

## chr3L
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chr3L.maf.gz
gzip -d chr3L.maf.gz
maf_parse -g $bed chr3L.maf >> $outfile'.maf'
rm chr3L.maf

## chr3R
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chr3R.maf.gz
gzip -d chr3R.maf.gz
maf_parse -g $bed chr3R.maf >> $outfile'.maf'
rm chr3R.maf

## chr4
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chr4.maf.gz
gzip -d chr4.maf.gz
maf_parse -g $bed chr4.maf >> $outfile'.maf'
rm chr4.maf

## chrX
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chrX.maf.gz
gzip -d chrX.maf.gz
maf_parse -g $bed chrX.maf >> $outfile'.maf'
rm chrX.maf

## chrY
wget https://hgdownload.soe.ucsc.edu/goldenPath/dm6/multiz27way/maf/chrY.maf.gz
gzip -d chrY.maf.gz
maf_parse -g $bed chrY.maf >> $outfile'.maf'
rm chrY.maf

# Call RNAz on the MAF file
RNAz -o $oufile'.rnaz.2s' $outfile'.maf'