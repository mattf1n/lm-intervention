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
    fields = ['model_version', 'do_filter', 'stat', 'mean_total_effect', 'mean_model_indirect_effect',
              'mean_model_direct_effect', 'mean_sum_indirect_effect', 'prop_aligned', 'num_examples_loaded',
              'num_examples_analyzed']
    t = PrettyTable(fields)
    for filter in ['filtered', 'unfiltered']:
        latex =  '\\hline\nModel & Param & TE (BLS) & TE (Bergsma) \\\\\n\\hline\n'
        for model_version in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large']:
            bergsma_effect = None
            bls_effect = None
            for stat in ['bergsma', 'bls']:
                fname =  f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
                    if stat == 'bergsma':
                        bergsma_effect = data['mean_total_effect']
                    else:
                        bls_effect =  data['mean_total_effect']
                    data['stat'] = stat
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
            latex += f'{model_to_name[model_version]} & {model_to_size[model_version]} & {bls_effect: .3f} &' \
                     f'{bergsma_effect: .3f} \\\\\n\\hline\n'
        with open(f'results/attention_intervention/winogender_{filter}_latex.txt', 'w') as o:
            o.write(latex)

    t.align = 'r'
    t.float_format = '.3'
    with open('results/attention_intervention/winogender_summary.txt', 'w') as o:
        o.write('WINOGENDER SUMMARY:\n\n')
        o.write(t.get_string())


if __name__ == '__main__':
    main()