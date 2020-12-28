#!/bin/bash

# ACTIVATE YOUR ENVIRONMENT HERE

python neuron_experiment_multiple_templates_num_agreement.py gpt2-xl cuda . false within_rc_singular_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py gpt2-xl cuda . false within_rc_plural_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py gpt2-xl cuda . false rc_plural_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py gpt2-xl cuda . false rc_singular_no_that 3 200
