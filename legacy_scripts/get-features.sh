#!/usr/bin/env bash

# Define the conditions for running the USCS programs as a bash function to enable concurrancy

function get_summary {
    track=$1
    url=$2
    bed=$3
    offset=$4
    output=$5
    if [[ "$url" = *.bb || "$url" = *.BigBed || "$url" = *.bigbed || "$url" = *.BB ]]; then
        chr=$(echo $bed | awk '{print $1}')
        start=$(echo $bed | awk '{print $2}')
        end=$(echo $bed | awk '{print $3}')
        name=$(echo $bed | awk '{print $4}')
        cov=$(bigBedSummary $url $chr $start $end 1 -type=coverage)
        mean=$(bigBedSummary $url $chr $start $end 1 -type=mean)
        min=$(bigBedSummary $url $chr $start $end 1 -type=min)
        max=$(bigBedSummary $url $chr $start $end 1 -type=max)
        echo -e $name'\t'$cov'\t'$mean'\t'$min'\t'$max >> $output/$track'.tsv'
    elif [[ "$track" = 'CAGE_pos' ]]; then
        chr=$(echo $bed | awk '{print $1}')
        start=$(echo $bed | awk '{print $2}')
        startNegOff=$(echo $start $offset | awk '{print $1 - $2}')
        startPosOff=$(echo $start $offset | awk '{print $1 + $2}')
        end=$(echo $bed | awk '{print $3}')
        endNegOff=$(echo $end $offset | awk '{print $1 - $2}')
        endPosOff=$(echo $end $offset | awk '{print $1 + $2}')
        name=$(echo $bed | awk '{print $4}')
        strd=$(echo $bed | awk '{print $6}')
        if [ "$strd" = '+' ]; then
            startPosTSS=$(bigWigSummary $url $chr $startNegOff $startPosOff 1 -type=max)
            echo -e $name'\t'$startPosTSS >> $output/$track'.tsv'
        elif [ "$strd" = '.' ]; then
            startPosTSS=$(bigWigSummary $url $chr $startNegOff $startPosOff 1 -type=max)
            echo -e $name'\t'$startPosTSS >> $output/$track'.tsv'
        fi
    elif [[ "$track" = 'CAGE_neg' ]]; then
            chr=$(echo $bed | awk '{print $1}')
            start=$(echo $bed | awk '{print $2}')
            startNegOff=$(echo $start $offset | awk '{print $1 - $2}')
            startPosOff=$(echo $start $offset | awk '{print $1 + $2}')
            end=$(echo $bed | awk '{print $3}')
            endNegOff=$(echo $end $offset | awk '{print $1 - $2}')
            endPosOff=$(echo $end $offset | awk '{print $1 + $2}')
            name=$(echo $bed | awk '{print $4}')
            strd=$(echo $bed | awk '{print $6}')
            if [ "$strd" = '-' ]; then
                endNegTSS=$(bigWigSummary $url $chr $endNegOff $endPosOff 1 -type=min)
                echo -e $name'\t'$endNegTSS >> $output/$track'.tsv'
            elif [ "$strd" = '.' ]; then
                endNegTSS=$(bigWigSummary $url $chr $endNegOff $endPosOff 1 -type=min)
                echo -e $name'\t'$endNegTSS >> $output/$track'.tsv'
            fi
    fi
}

function get_avg_over_bed {
    track=$1
    url=$2
    bed=$3
    if [[ "$url" = *.bw && "$track" != CAGE* ]]; then
        bigWigAverageOverBed $url $bed $track'.tsv' -minMax
    elif [[ "$track" = 'CAGE_pos' ]]; then
        bigWigAverageOverBed $url $bed $track'_whole_trans.tsv' -minMax
    elif [[ "$track" = 'CAGE_neg' ]]; then
        bigWigAverageOverBed $url $bed $track'_whole_trans.tsv' -minMax
    fi
}

export -f get_summary
export -f get_avg_over_bed

feature=$1
bed=$(readlink -f $2)
#outfile=$(echo ${bed%.*} | awk -F'[/]' '{print$(NF)}')
workdir=$(readlink -f $3)
output=$workdir/features/
downstream_threads=$4

# offset is the number of basepairs up and downstream of the transcript beguining (variables start or end depending on strand + or -).
if [ -z $offset ]; then
    offset=100
else
    offset=$5
fi

mkdir -p $output 
cd $output

track=$(echo $feature | awk -F '[ ]' '{print $(1)}')
url=$(echo $feature | awk -F '[ ]' '{print $(2)}')

if [[ "$url" = *.bw ]]; then
    get_avg_over_bed $track $url $bed
    parallel --no-notice -k --lb -j $downstream_threads -a $bed get_summary $track $url {} $offset $output
else 
    parallel --no-notice -k --lb -j $downstream_threads -a $bed get_summary $track $url {} $offset $output
fi
