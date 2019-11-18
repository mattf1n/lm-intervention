import json
from prettytable import PrettyTable
import pandas as pd
import numpy as np


def main():
    fields = ['model_version', 'do_filter', 'stat', 'mean_total_effect', 'mean_model_indirect_effect',
              'mean_model_direct_effect', 'mean_sum_indirect_effect', 'prop_aligned', 'num_examples_loaded',
              'num_examples_analyzed']
    t = PrettyTable(fields)
    for filter in ['filtered', 'unfiltered']:
        for model_version in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
            for stat in ['bergsma', 'bls']:
                fname =  f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
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

    t.align = 'r'
    t.float_format = '.3'
    with open('results/attention_intervention/winogender_summary.txt', 'w') as o:
        o.write('WINOGENDER SUMMARY:\n\n')
        o.write(t.get_string())


if __name__ == '__main__':
    main()