#!/usr/bin/env bash

## VARIABLES ##
workdir=$1
sra=$2
threads=$3
appdir=$4
metadata=$5

if [[ -z ${bed+x} ]]; then
  bed=$workdir/results/new-non-coding.chr.bed
  USER_PREDICT=0
else
  bed=$(readlink -f $6) &>> $workdir/run.log
  USER_PREDICT=1
  PIPE_STEP=7
fi

#genome=$(readlink -f $6)
#annot=$(readlink -f $7)

## COLORS ##
BOLD='\033[1m'
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
PURPLE='\033[1;35m'
NC='\033[0m'

mkdir -p $workdir
touch "$workdir"/run.log

## DEFINE THREADS FOR RUUNING PIPELINE ##
jobs=$(wc -l $sra | cut -f1 -d' ')

echo 'Threads available: ' $threads &>> $workdir/run.log

echo 'Number of samples for this run: ' $jobs &>> $workdir/run.log
if [[ $threads -ge $jobs ]]; then
    downstream_threads=$(expr $threads / $jobs)
    downstream_threads=${downstream_threads%.*}
else
    jobs=$threads
    downstream_threads=1
fi
echo 'Number of threads per running sample: ' $downstream_threads &>> $workdir/run.log


## ANIMATED OUTPUT ##
# Load in the functions and animations
source "$appdir"/scripts/anime.sh
# Run BLA::stop_loading_animation if the script is interrupted
trap BLA::stop_loading_animation SIGINT

# BLA::start_loading_animation "${BLA_braille_whitespace[@]}"
# BLA::stop_loading_animation

conda_path=$(conda info | grep -i 'base environment' | awk '{print$(4)}')
conda_sh=$conda_path'/etc/profile.d/conda.sh'

source $conda_sh
conda init $(echo $SHELL | awk -F'[/]' '{print$(NF)}') &> $appdir/cmd.out

## INITIATE PIPELINE ##
PIPE_STEP=1
BLA::start_loading_animation "${BLA_braille_whitespace[@]}"
while true;
do
  if [[ PIPE_STEP -eq 0 ]]; then
    BLA::stop_loading_animation
    echo -e "\r\e[9A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${GREEN}[🧩] Mapping samples to reference genome${NC}
${GREEN}[🧱] Building transcriptomes${NC}
${GREEN}[🎲] Calculating conding probability of new transcripts${NC}
${GREEN}[📈] Differential transcript expressions analysis (if -m)${NC}
${GREEN}[📡] Extracting candidate features from databases${NC}"
    echo -e "${YELLOW}Program terminated. If you ran into any errors check${NC} run.log ${YELLOW}on the output directory${NC}"
    break
  elif [[ PIPE_STEP -eq 1 ]]; then
    echo -e "\r[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${CYAN}[-] Preparing reference genome files${NC}
${CYAN}[ ] Gathering sample information from SRA database {flync sra}${NC}
${CYAN}[ ] Mapping samples to reference genome${NC}
${CYAN}[ ] Building transcriptomes${NC}
${CYAN}[ ] Calculating conding probability of new transcripts${NC}
${CYAN}[ ] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 2 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${CYAN}[-] Gathering sample information from SRA database {flync sra}${NC}
${CYAN}[ ] Mapping samples to reference genome${NC}
${CYAN}[ ] Building transcriptomes${NC}
${CYAN}[ ] Calculating conding probability of new transcripts${NC}
${CYAN}[ ] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 3 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${CYAN}[-] Mapping samples to reference genome${NC}
${CYAN}[ ] Building transcriptomes${NC}
${CYAN}[ ] Calculating conding probability of new transcripts${NC}
${CYAN}[ ] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 4 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${GREEN}[🧩] Mapping samples to reference genome${NC}
${CYAN}[-] Building transcriptomes${NC}
${CYAN}[ ] Calculating conding probability of new transcripts${NC}
${CYAN}[ ] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 5 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${GREEN}[🧩] Mapping samples to reference genome${NC}
${GREEN}[🧱] Building transcriptomes${NC}
${CYAN}[-] Calculating conding probability of new transcripts${NC}
${CYAN}[ ] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 6 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${GREEN}[🧩] Mapping samples to reference genome${NC}
${GREEN}[🧱] Building transcriptomes${NC}
${GREEN}[🎲] Calculating conding probability of new transcripts${NC}
${CYAN}[-] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[ ] Extracting candidate features from databases${NC}"
  elif [[ PIPE_STEP -eq 7 ]]; then
    echo -e "\r\e[8A\e[K[🦟] ${BOLD}${PURPLE}FLYNC is processing your samples:${NC}
${GREEN}[🧬] Preparing reference genome files${NC}
${GREEN}[📑] Gathering sample information from SRA database {flync sra}${NC}
${GREEN}[🧩] Mapping samples to reference genome${NC}
${GREEN}[🧱] Building transcriptomes${NC}
${GREEN}[🎲] Calculating conding probability of new transcripts${NC}
${GREEN}[📈] Differential transcript expressions analysis (if -m)${NC}
${CYAN}[-] Extracting candidate features from databases${NC}"
  fi

  case $PIPE_STEP in
    exit) break ;;
    1)
      ## RUN SCRIPTS FOR GETTING GENOME AND INFO ON SRA RUNS
      conda activate infoMod &>> $workdir/run.log

      ## SILENCE PARALLEL FIRST RUN ##
      parallel --citation &> $appdir/cmd.out
      echo will cite &> $appdir/cmd.out
      rm $appdir/cmd.out

      mkdir -p $appdir/genome &>> $workdir/run.log
      $appdir/scripts/get-genome.sh $appdir &>> $workdir/run.log
      PIPE_STEP=2
      ;;
    2)
      mkdir -p $workdir/results
      $appdir/scripts/get-sra-info.sh $workdir $sra &>> $workdir/run.log
      PIPE_STEP=3
      conda deactivate
      ;;
    3)
      conda activate mapMod &>> $workdir/run.log
      $appdir/scripts/build-index.sh $appdir $threads &>> $workdir/run.log
      parallel -k --lb -j $jobs -a $sra $appdir/scripts/tux2map.sh $workdir {} $downstream_threads $appdir &>> $workdir/run.log
      PIPE_STEP=4
      conda deactivate
      ;;
    4)
      conda activate assembleMod &>> $workdir/run.log
      parallel -k --lb -j $jobs -a $sra $appdir/scripts/tux2assemble.sh $workdir {} $downstream_threads $appdir &>> $workdir/run.log
      $appdir/scripts/tux2merge.sh $workdir $sra $threads $appdir &>> $workdir/run.log
      parallel -k --lb -j $jobs -a $sra $appdir/scripts/tux2count.sh $workdir {} &>> $workdir/run.log
      PIPE_STEP=5
      conda deactivate
      ;;
    5)
      conda activate codMod &>> $workdir/run.log
      $appdir/scripts/coding-prob.sh $workdir $appdir $threads &>> $workdir/run.log
      $appdir/scripts/class-new-transfrags.sh $workdir $threads $appdir &>> $workdir/run.log
      PIPE_STEP=6
      conda deactivate
      ;;
    6)
      conda activate dgeMod
      # ballgown for dge analysis
      if [[ -z ${metadata+x} ]]; then
          echo "Skipping DGE since no metadata file was provided..." &>> $workdir/run.log
      else
          Rscript $appdir/scripts/ballgown.R $(readlink -f $workdir) $(readlink -f $metadata) &>> $workdir/run.log
      fi

      PIPE_STEP=7
      conda deactivate
      ;;
    7)
      conda activate infoMod
      mkdir -p $workdir/results
      cd $workdir/results
      $appdir/scripts/gtf-to-bed.sh new-non-coding.gtf &>> $workdir/run.log
      conda deactivate

      cd $workdir
      conda activate featureMod
      
      mkdir -p $workdir/results/non-coding/features
      
      # Got 24% faster by parallelizing the get-features.sh script
      jobs2=$(cat $appdir/static/tracksFile.tsv | wc -l)
      downstream_threads=$(expr $threads / $jobs2)
      downstream_threads=${downstream_threads%.*}
      parallel --no-notice -k --lb -j $jobs2 -a $appdir/static/tracksFile.tsv $appdir/scripts/get-features.sh {} $bed $workdir/results/non-coding $downstream_threads &>> $workdir/run.log

      # Write a .csv file with the filepaths for the tables to be processed in python Pandas
      ls $workdir/results/non-coding/features | grep tsv | sed 's/.tsv//g' > names.tmp
      find $workdir/results/non-coding/features/*.tsv > path.tmp
      paste -d, names.tmp path.tmp > $workdir/results/non-coding/features/paths.csv
      rm names.tmp path.tmp &>> $workdir/run.log

      conda deactivate

      ### TO-DO: TRANSFER THIS TO NEW STEP ###

      conda activate predictMod

      python3 $appdir/scripts/feature-table.py $appdir $workdir $bed &>> $workdir/run.log

      if [[ $USER_PREDICT = 0 ]]; then
        python3 $appdir/scripts/predict.py $appdir $workdir &>> $workdir/run.log
        python3 $appdir/scripts/final-table.py $appdir $workdir $bed &>> $workdir/run.log
      else
        python3 $appdir/scripts/predict.py $appdir $workdir "$(basename $bed | awk -F'[.]' '{print$1}')" &>> $workdir/run.log
      fi
      
      PIPE_STEP=0
      ;;
  esac
done
