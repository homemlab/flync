#!/usr/bin/env bash

feature=$1
bed=$(readlink -f $2)
outfile=$(echo $2 | awk -F'.' '{print $(NF-2)}')
workdir=$(readlink -f $3)
output=$workdir/features/$outfile

# offset is the number of basepairs up and downstream of the transcript beguining (variables start or end depending on strand + or -).
if [ -z $offset ]; then
    offset=100
else
    offset=$4
fi

mkdir -p $output 
cd $output

track=$(echo $feature | awk -F '[ ]' '{print $(1)}')
url=$(echo $feature | awk -F '[ ]' '{print $(2)}')

if [[ "$url" == *.bw && "$track" != CAGE* ]]; then
    bigWigAverageOverBed $url $bed $track'.tsv' -minMax
elif [[ "$url" == *.bb || "$url" == *.BigBed || "$url" == *.bigbed || "$url" == *.BB ]]; then
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
        echo -e $name'\t'$cov'\t'$mean'\t'$min'\t'$max >> $track'.tsv'
    done < $bed
elif [[ "$track" == 'CAGE_pos' ]]; then
    bigWigAverageOverBed $url $bed $track'_whole_trans.tsv' -minMax
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
            echo -e $name'\t'$startPosTSS >> $track'.tsv'
        elif [ $strd == '.' ]; then
            startPosTSS=$(bigWigSummary $url $chr $startNegOff $startPosOff 1 -type=max)
            echo -e $name'\t'$startPosTSS >> $track'.tsv'
        fi
    done < $bed
elif [[ "$track" == 'CAGE_neg' ]]; then
    bigWigAverageOverBed $url $bed $track'_whole_trans.tsv' -minMax
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
            echo -e $name'\t'$endNegTSS >> $track'.tsv'
        elif [ $strd == '.' ]; then
            endNegTSS=$(bigWigSummary $url $chr $endNegOff $endPosOff 1 -type=min)
            echo -e $name'\t'$endNegTSS >> $track'.tsv'
        fi
    done < $bed
fi
