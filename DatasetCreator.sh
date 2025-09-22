#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=26
#SBATCH --time=00:20:00
#SBATCH --partition=amilan
#SBATCH --account=ucb530_asc1
#SBATCH --output=output.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mawa5935@colorado.edu
#SBATCH --job-name=DatasetCreator

module purge

module load python
module load anaconda

conda activate EEG2TEXT

echo "== Running Script! =="
python Split.py
echo "== End of Job =="
