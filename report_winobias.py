import glob
import json
from prettytable import PrettyTable


def main():
    fields = ['model_version', 'split', 'do_filter', 'mean_total_effect', 'num_examples_loaded',
              'num_examples_analyzed', 'filter_quantile', 'threshold', 'num_examples_aligned']
    t = PrettyTable(fields)
    for fname in glob.glob('winobias_data/*.json'):
        with open(fname) as f:
            data = json.load(f)
            try:
                t.add_row([data.get(field, '') for field in fields])
            except (KeyError, AttributeError):
                print('Skipping file:', fname)
    print("WINOBIAS RESULTS:")
    print(t)


if __name__ == '__main__':
    main()