#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 0-06:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_dgx1        # Partition to submit to
#SBATCH --mem=16000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1        # Number of gpus
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/2019.10
source activate tf2
# python winobias_attn_intervention.py --gpt2-version transfo-xl-wt103 --do-filter True --split dev
python winogender_attn_intervention.py --gpt2-version transfo-xl-wt103 --do-filter True --stat bergsma
