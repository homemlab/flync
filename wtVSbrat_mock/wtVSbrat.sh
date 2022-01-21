#!/bin/bash

#SBATCH --job-name=wtvsbrat
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48

# Be sure to request the correct partition to avoid the job to be held in the queue, furthermore
#	on CIRRUS-B (Minho)  choose for example HPC_4_Days
#	on CIRRUS-A (Lisbon) choose for example hpc
#SBATCH --partition=hpc

# Used to guarantee that the environment does not have any other loaded module
module purge

source activate
conda activate dlinct-ml-dev3

# Execute commands or scripts
./parallel.sh $PWD/wtVSbrat_mock runlist_wtVSbrat_Landskron2018.txt 48 $PWD metadata_wtVSbrat_Landskron2018.csv $PWD/genome.fa $PWD/mock.gtf

echo "Finished with job $SLURM_JOBID"