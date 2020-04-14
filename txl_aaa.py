import csv
import pandas as pd

# file_path = ('/lm-intervention/txl_results/neuron_intervention/results/' +
#              '20200412_neuron_intervention/transfo-xl-wt103_neuron_effects.csv')

# with open(file_path) as csvfile:
#     reader = csv.reader(csvfile)
#     for i, row in enumerate(reader):
#         print(row)
#         print()
#         if i >= 2: break

file_path = ('~/lm-intervention/txl_results/neuron_intervention/results/' +
             '20200412_neuron_intervention/The_X_said_that_man_direct_transfo-xl-wt103.csv')

pd.set_option('display.max_columns', 500)
df = pd.read_csv(file_path)
print(df.head(5))
# print(df[1020:1030])
