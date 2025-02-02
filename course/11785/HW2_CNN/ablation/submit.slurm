#!/bin/bash
#SBATCH -J test
#SBATCH -A m3578
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q regular
#SBATCH --exclusive
# #SBATCH -t 00:00:00
#SBATCH -t 00:10:00

module load python
conda activate gator

python /global/cfs/cdirs/m3578/jiayihua/11785/convnext.py 
