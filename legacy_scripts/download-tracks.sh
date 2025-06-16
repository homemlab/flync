#!/usr/bin/env bash


conda activate mapMod

parallel --citation &> /dev/null && echo "will cite"

mkdir -p /bin/app/static/tracks

</bin/app/static/tracksFile.tsv | awk '{print $2}' | parallel -j 4 wget -q -P /bin/app/static/tracks/ {}

export TRACKS_DOWNLOAD_STATUS=true
