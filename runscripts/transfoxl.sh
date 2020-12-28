#!/bin/bash

# ACTIVATE YOUR ENVIRONMENT HERE

python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false none 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false within_rc_singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false within_rc_plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false distractor 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false distractor_1 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false rc_singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false rc_plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false singular 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false plural 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false rc_singular_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false rc_plural_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false within_rc_singular_no_that 3 200
python neuron_experiment_multiple_templates_num_agreement.py transfo-xl-wt103 cuda . false within_rc_plural_no_that 3 200
