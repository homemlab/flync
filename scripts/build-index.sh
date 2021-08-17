#bin/bash

appdir=$1
threads=$2
cd $appdir
mkdir genome &> /dev/null
cd genome
export HISAT2_INDEXES=$appdir/genome/

### Build genome index '('One-time-only command')'

idx=$(ls | grep -c '.ht2$')

if [ $idx == 0 ]
then
    echo ----- BUILDING GENOME INDEX -----
    hisat2-build -p $threads $appdir/genome/dm6.fa dm6.idx

    echo ----- EXTRACTING SPLICE JUNCTIONS -----
    hisat2_extract_splice_sites.py $appdir/genome/dm6.gtf > dm6.ss
fi