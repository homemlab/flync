#!/bin/bash

#SBATCH --job-name=L3_CNS
#SBATCH --mem=69G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --partition=hpc

# Used to guarantee that the environment does not have any other loaded module
module purge

source activate
conda activate dlinct-ml-dev3

# Execute commands or scripts
./parallel.sh $PWD/L3_CNS runlist_L3_CNS_modENCODEandFlyatlas2.txt $SLURM_NTASKS $PWD

echo "Finished with job $SLURM_JOBID"
