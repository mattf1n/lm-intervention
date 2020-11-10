
models = [
        {
            'name': 'distilgpt2',
            'mem': 16000,
            'time': '0-03:00',},
        {
            'name': 'gpt2',
            'mem': 16000,
            'time': '0-06:00',},
        {
            'name': 'gpt2-medium', 
            'mem': 16000,
            'time': '0-12:00',},
        {
            'name': 'gpt2-large',
            'mem': 100000, 
            'time': '1-00:00',},
        {
            'name': 'gpt2-xl',
            'mem': 100000, 
            'time': '1-00:00',},]

examples = [
        'none',
        'singular',
        'plural',
        'rc_singular', 
        'rc_plural', 
        'within_rc_singular', 
        'within_rc_plural', 
        'distractor',
        'distractor_1',]

for random in ['random', 'trained']:
    for model in models:
        for example in examples:
            script = f'''#!/bin/bash
#SBATCH -n 1                # Number of cores
#SBATCH -N 1                # Ensure that all cores are on one machine
#SBATCH -t {model['time']}          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_dgx1        # Partition to submit to
#SBATCH --mem={model['mem']}         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --gres=gpu:1        # Number of gpus
#SBATCH -o outputs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e outputs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
python neuron_experiment_multiple_templates_num_agreement.py {model['name']} cuda ./ false {example} 3 200
'''
            with open(
                    'runscripts/'
                    + random 
                    + '/' 
                    + '_'.join([model['name'], example, random,]) 
                    + '.sh',
                    'w') as f:
                f.write(script)



