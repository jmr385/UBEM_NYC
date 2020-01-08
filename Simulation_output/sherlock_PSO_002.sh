#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<jmroth>@stanford.edu
#SBATCH --job-name=batch65
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --time=50:00
#SBATCH --output=pso.log
# load modules
ml python/3.6.1
pip install --user -r requirements2.txt
ml py-numpy/1.17.2_py36

# execute script
python3 ubem_simulation_pso.py