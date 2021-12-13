#!/bin/bash

tracksfile=$(readlink -f $1)
bed=$(readlink -f $2)
output=$3

# offset is the number of basepairs up and downstream of the transcript beguining (variables start or end depending on strand + or -).
if [ -z $offset ]; then
    offset=100
else
    offset=$4
fi

mkdir $output &> $workdir

while read i
do
    track=$(echo $i | awk -F '[ ]' '{print $(1)}')
    echo $track
    url=$(echo $i | awk -F '[ ]' '{print $(2)}')
    echo $url
    if [[ "$url" == *.bw && "$track" != CAGE* ]]; then
        bigWigAverageOverBed $url $bed $output/$track'.tsv' -minMax
    elif [[ "$url" == *.bb ]]; then
        while read s
        do
            chr=$(echo $s | cut -f1 -d' ')
            start=$(echo $s | cut -f2 -d' ')
            end=$(echo $s | cut -f3 -d' ')
            name=$(echo $s | cut -f4 -d' ')
            cov=$(bigBedSummary $url $chr $start $end 1 -type=coverage)
            mean=$(bigBedSummary $url $chr $start $end 1 -type=mean)
            min=$(bigBedSummary $url $chr $start $end 1 -type=min)
            max=$(bigBedSummary $url $chr $start $end 1 -type=max)
            echo -e $name'\t'$cov'\t'$mean'\t'$min'\t'$max >> $output/$track'.tsv'
        done < $bed
    elif [[ "$track" == 'CAGE_pos' ]]; then
        bigWigAverageOverBed $url $bed $output/$track'_whole_trans.tsv' -minMax
        while read s
        do
            chr=$(echo $s | cut -f1 -d' ')
            start=$(echo $s | cut -f2 -d' ')
            startNegOff=$(expr $start - $offset)
            startPosOff=$(expr $start + $offset)
            end=$(echo $s | cut -f3 -d' ')
            endNegOff=$(expr $end - $offset)
            endPosOff=$(expr $end + $offset)
            end=$(echo $s | cut -f3 -d' ')
            name=$(echo $s | cut -f4 -d' ')
            strd=$(echo $s | cut -f6 -d ' ')
            if [ $strd == '+' ]; then
                startPosTSS=$(bigWigSummary $url $chr $startNegOff $startPosOff 1 -type=max)
                echo -e $name'\t'$startPosTSS >> $output/$track'.tsv'
            elif [ $strd == '.' ]; then
                startPosTSS=$(bigWigSummary $url $chr $startNegOff $startPosOff 1 -type=max)
                echo -e $name'\t'$startPosTSS >> $output/$track'.tsv'
            fi
        done < $bed
    elif [[ "$track" == 'CAGE_neg' ]]; then
        bigWigAverageOverBed $url $bed $output/$track'_whole_trans.tsv' -minMax
        while read s
        do
            chr=$(echo $s | cut -f1 -d' ')
            start=$(echo $s | cut -f2 -d' ')
            startNegOff=$(expr $start - $offset)
            startPosOff=$(expr $start + $offset)
            end=$(echo $s | cut -f3 -d' ')
            endNegOff=$(expr $end - $offset)
            endPosOff=$(expr $end + $offset)
            name=$(echo $s | cut -f4 -d' ')
            strd=$(echo $s | cut -f6 -d ' ')
            if [ $strd == '-' ]; then
                endNegTSS=$(bigWigSummary $url $chr $endNegOff $endPosOff 1 -type=min)
                echo -e $name'\t'$endNegTSS >> $output/$track'.tsv'
            elif [ $strd == '.' ]; then
                endNegTSS=$(bigWigSummary $url $chr $endNegOff $endPosOff 1 -type=min)
                echo -e $name'\t'$endNegTSS >> $output/$track'.tsv'
            fi
        done < $bed
    fi
done < $tracksfile

# Write a .csv file with the filepaths for the tables to be processed in python Pandas
ls $output | grep tsv | sed 's/.tsv//g' > names.tmp
find $output/*.tsv > path.tmp
paste names.tmp path.tmp > $output/paths.txt
rm names.tmp path.tmp