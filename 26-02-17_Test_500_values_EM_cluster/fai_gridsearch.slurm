#!/bin/bash
#
#SBATCH --cluster=wice
#
#SBATCH --job-name='grid_search_fai'
#
#SBATCH --nodes=2
#
#SBATCH --ntasks-per-node=72
#
#SBATCH --ntasks=144
#
#SBATCH -t 04:30:00
#
#SBATCH -A lp_cmpa
#
#SBATCH --mail-type="BEGIN,END,FAIL"
#
#SBATCH --mail-user=paulajessica.gonzalezprieto@student.kuleuven.be   
#

echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_NODELIST"
echo "Start time: $(date)"
echo "======================================="
START_TIME=$(date +%s)

# 1️⃣ Limpiar módulos
module purge

source /user/leuven/386/vsc38664/miniforge3/bin/activate
conda activate sunpy

echo " "
which python

echo "Iniciando grid search de FAI..."

# 4️⃣ Ejecutar script
python /scratch/leuven/386/vsc38664/Thesis/26-02-17_Data/grid_search_fai_parallel_EM.py

echo "Script terminado."

# Guardar tiempo final
END_TIME=$(date +%s)
echo "======================================="
echo "End time: $(date)"
echo "Total runtime: $((END_TIME - START_TIME)) seconds"
echo "======================================="
echo " "
