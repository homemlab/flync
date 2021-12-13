#!/bin/bash

appdir=$1
temp=$2
output=$3
bed=$(readlink -f $4)

mkdir $output >&2 $workdir/err.log

if [ -z $temp ]; then
    temp=37
fi

cd $output

if [ ! -e $output/seq_to_strct.fa ]; then
    bedtools getfasta -s -nameOnly -fi $appdir/genome/genome.fa -bed $bed -fo $output/seq_to_strct.fa
    # gffread -w $output/seq_to_strct.fa -g $appdir/genome/genome.fa $output/seq_to_strct.gtf
fi

#Run RNAfold on newly discovered sequences

while read i
do 
    if [ $(echo $i | grep '>' | wc -l) == 1 ]; then
        echo $i > seq.fa
    else echo $i >> seq.fa
        RNAfold -T $temp --noPS -i seq.fa >> $output/RNAfold.2s
    fi
done < $output/seq_to_strct.fa
wait

cat $output/RNAfold.2s | grep '>' | sed 's/>//' | sed 's/([^~]*)//g' > names.tmp
cat $output/RNAfold.2s | grep ' ' | awk -F '[-]' '{print "-"$(2)}' | sed 's/)//g' > mfe.tmp

wait
## Prepare RNAfold output table
set names
while read i
do
    names+=("$i")
done < names.tmp

set mfe
while read i
do
    mfe+=("$i")
done < mfe.tmp

wait
paste names.tmp mfe.tmp > $output/RNAfold.tsv
rm ./*_ss.ps names.tmp mfe.tmp


### MXfold not working... Going to move on with just RNAfold results

# Run mxfold on newly discovered sequences
# while read i
# do 
#     if [ $(echo $i | grep '>' | wc -l) == 1 ]; then
#         echo $i > seq.fa
#     else echo $i >> seq.fa
#         mxfold seq.fa >> mxfold.2s
#     fi
# done < $output/seq_to_strct.fa

# cat mxfold.2s | grep -v '>structure' > mxfold.2s.tmp

# rm mxfold.2s seq.fa
# mv mxfold.2s.tmp mxfold.2s

# # Use RNAeval to get MFE of mxfold strcutures
# RNAeval -T $temp mxfold.2s > mxmfe.2s
# cat $output/mxmfe.2s | grep ' ' | awk -F '[ ]' '{print $(2)}' | awk -F '[(]' '{print $(2)}' | awk -F '[)]' '{print $(NF-1)}' > mxmfe.tmp

# paste names.tmp mxmfe.tmp > $output/mxfold.tsv
# rm names.tmp mfe.tmp mxmfe.tmp 