#!/bin/bash
#SBATCH --job-name=patches
#SBATCH --output=patches.out
#SBATCH --error=patches.err
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --constraint=cpu
#SBATCH --mail-user=tua_mail@...
#SBATCH --mail-type=ALL

module load python
conda activate tuo_ambiente_conda

ntask=64
nnode=$SLURM_JOB_NUM_NODES
let ntask_node=$ntask/$nnode


echo "Start: " `date`
echo "Running with"
echo "            nnode = ${nnode}"
echo "            ntask = ${ntask}"
echo "       ntask_node = ${ntask_node}"


module load python
conda activate tuo_ambiente_conda
srun -n $ntask \
python /global/homes/m/touaccount/....../script.py

