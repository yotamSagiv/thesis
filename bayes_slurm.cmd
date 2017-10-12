#!/usr/bin/env bash

#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1
#SBATCH -t 24:00:00
#SBATCH --mail-type=begin 
#SBATCH --mail-type=end 
#SBATCH --mail-user=ysagiv@princeton.edu 

module load anaconda
source activate thesis
python explore_parameters.py
source deactivate