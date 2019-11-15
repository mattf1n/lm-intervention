import json
from prettytable import PrettyTable
import pandas as pd
import numpy as np

model_to_size = {
    'distilgpt2': '-37\\%',
    'gpt2': '124M',
    'gpt2-medium': '355M',
    'gpt2-large': '774M'
}
model_to_name = {
    'distilgpt2': 'Distill-GPT2',
    'gpt2': 'GPT2-small',
    'gpt2-medium': 'GPT2-medium',
    'gpt2-large': 'GPT2-large'
}

def main():
    fields = ['model_version', 'do_filter', 'split', 'mean_total_effect', 'mean_model_indirect_effect',
              'mean_model_direct_effect', 'mean_sum_indirect_effect', 'prop_aligned', 'num_examples_loaded',
              'num_examples_analyzed']
    t = PrettyTable(fields)
    for filter in ['filtered', 'unfiltered']:
        latex =  '\\hline\nModel & Param & TE (dev) & TE (test) \\\\\n\\hline\n'
        for model_version in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large']:
            dev_effect = None
            test_effect = None
            for split in ['dev', 'test']:
                fname =  f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                with open(fname) as f:
                    data = json.load(f)
                    if split == 'dev':
                        dev_effect = data['mean_total_effect']
                    else:
                        test_effect = data['mean_total_effect']
                    # Populate 'prop_aligned'
                    try:
                        data['prop_aligned'] = data['num_examples_aligned'] / data['num_examples_loaded']
                    except KeyError:
                        data['prop_aligned'] = ''
                    # Populate mean_sum_indirect
                    results = data['results']
                    df = pd.DataFrame(results)
                    # Convert to shape (num_examples X num_layers X num_heads)
                    indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
                    data['mean_sum_indirect_effect'] = indirect_by_head.sum(axis=(1, 2)).mean()
                    try:
                        t.add_row([data.get(field, '') for field in fields])
                    except (KeyError, AttributeError):
                        print('Skipping file:', fname)
            latex += f'{model_to_name[model_version]} & {model_to_size[model_version]} & {dev_effect: .3f} &' \
                     f'{test_effect: .3f} \\\\\n\\hline\n'
        with open(f'results/attention_intervention/winobias_{filter}_latex.txt', 'w') as o:
            o.write(latex)

    t.align = 'r'
    t.float_format = '.3'
    with open('results/attention_intervention/winobias_summary.txt', 'w') as o:
        o.write('WINOBIAS SUMMARY:\n\n')
        o.write(t.get_string())


if __name__ == '__main__':
    main()