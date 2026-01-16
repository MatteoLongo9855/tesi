### riga da terminale per allocare nodo interattivo
### salloc -N1 -t 00:30:00 -q interactive -C cpu


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

