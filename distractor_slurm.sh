#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_dgx1        # Partition to submit to
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1        # Number of gpus
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load Anaconda3/2019.10
source activate myenv
python neuron_experiment_multiple_templates_num_agreement.py disgtilgpt2 cuda ./ false distractor 5 30
python neuron_experiment_multiple_templates_num_agreement.py gpt2 cuda ./ false distractor 5 30
python neuron_experiment_multiple_templates_num_agreement.py gpt2-medium cuda ./ false distractor 5 30
python neuron_experiment_multiple_templates_num_agreement.py gpt2-large cuda ./ false distractor 5 30
python neuron_experiment_multiple_templates_num_agreement.py gpt2-xl cuda ./ false distractor 5 30
