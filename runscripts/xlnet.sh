#!/bin/bash

# ACTIVATE YOUR ENVIRONMENT HERE
#SBATCH -n 1                # Number of cores 
#SBATCH -N 1                # Ensure that all cores are on one machine 
#SBATCH -t 1-00:00          # Runtime in D-HH:MM, minimum of 10 minutes 
#SBATCH -p seas_dgx1        # Partition to submit to 
#SBATCH --mem=100000         # Memory pool for all cores (see also --mem-per-cpu) 
#SBATCH --gres=gpu:1        # Number of gpus 
#SBATCH -o outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid 
#SBATCH -e outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid 

python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false none 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false within_rc_singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false within_rc_plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false distractor 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false distractor_1 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false rc_singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false rc_plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false rc_singular_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false rc_plural_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false within_rc_singular_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py xlnet-base-cased cuda . false within_rc_plural_no_that 3 200
