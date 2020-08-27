#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_dgx1        # Partition to submit to
#SBATCH --mem=300000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/2019.10
source activate myenv
python aggregate_total_effect_bar_plot.py results/na_neuron_intervention/ figures/ 
