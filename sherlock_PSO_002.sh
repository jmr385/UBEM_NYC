#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<jmroth>@stanford.edu
#SBATCH --job-name=PSO_sim500_BC1000_HC1000
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=PSO_sim500_BC1000_HC1000.log
# load modules
ml python/3.6.1
pip install --user -r requirements2.txt
ml py-numpy/1.17.2_py36

# execute script
python3 ubem_simulation_pso.py