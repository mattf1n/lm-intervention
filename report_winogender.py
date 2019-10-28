import json
from prettytable import PrettyTable


def main():
    fields = ['model_version', 'do_filter', 'stat', 'mean_total_effect', 'prop_aligned', 'num_examples_loaded',
              'num_examples_analyzed', 'filter_quantile', 'threshold', 'num_examples_aligned']
    t = PrettyTable(fields)
    for model_version in ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large']:
        for filter in ['filtered', 'unfiltered']:
            for stat in ['bergsma', 'bls']:
                fname =  f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                print(fname)
                with open(fname) as f:
                    data = json.load(f)
                    data['stat'] = stat
                    try:
                        data['prop_aligned'] = data['num_examples_aligned'] / data['num_examples_loaded']
                    except KeyError:
                        data['prop_aligned'] = ''
                    try:
                        t.add_row([data.get(field, '') for field in fields])
                    except (KeyError, AttributeError):
                        print('Skipping file:', fname)
    print("WINOGENDER RESULTS:")
    t.align = 'r'
    t.float_format = '.3'
    print(t)


if __name__ == '__main__':
    main()