#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<jmroth>@stanford.edu
#SBATCH --job-name=OOS_sim500_1000H_1000B_2
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --mem=20G
#SBATCH --time=2:15:00
#SBATCH --output=OOS_sim500_1000H_1000B_2.log
# load modules
ml python/3.6.1
pip install --user -r requirements2.txt
ml py-numpy/1.17.2_py36

# execute script
python3 ubem_simulation_pso.py