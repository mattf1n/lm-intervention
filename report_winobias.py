import json
from prettytable import PrettyTable
import pandas as pd
import numpy as np

def main():
    fields = ['model_version', 'do_filter', 'split', 'mean_total_effect', 'mean_sum_indirect_effect', 'prop_aligned', 'num_examples_loaded',
              'num_examples_analyzed', 'filter_quantile', 'threshold', 'num_examples_aligned']
    t = PrettyTable(fields)
    for model_version in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large']:
        for filter in ['filtered', 'unfiltered']:
            for split in ['dev', 'test']:
                fname =  f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                with open(fname) as f:
                    data = json.load(f)
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
    t.align = 'r'
    t.float_format = '.3'
    with open('results/attention_intervention/winobias_summary.txt', 'w') as o:
        o.write('WINOBIAS SUMMARY:\n\n')
        o.write(t.get_string())


if __name__ == '__main__':
    main()