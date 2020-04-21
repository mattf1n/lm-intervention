import pandas as pd
import os


def analyze_effect_results(file_path, alt):
    results_df = pd.read_csv(file_path)
    print(results_df[1:2].to_string()); assert False
    if alt == 'man':
        odds_base = results_df['candidate1_base_prob'] / results_df['candidate2_base_prob']
        odds_intervention = results_df['candidate1_prob'] / results_df['candidate2_prob']
    else:
        odds_base = results_df['candidate2_base_prob'] / results_df['candidate1_base_prob']
        odds_intervention = results_df['candidate2_prob'] / results_df['candidate1_prob']
    odds_ratio = odds_intervention / odds_base
    results_df['odds_ratio'] = odds_ratio
    results_df = results_df.groupby(['layer', 'neuron'], as_index=False).mean()
    # results_df.to_csv('txl_neuron_analysis/txl_analyze_' + file_path.split('/')[-1])


folder_name = '/n/shieber_lab/Lab/users/ssakenis/txl_neuron_results/'

fnames = [f for f in os.listdir(folder_name)]
paths = [os.path.join(folder_name, f) for f in fnames]
woman_ind_files = [f for f in paths if '_woman_indirect' in f]
man_ind_files = [f for f in paths if '_man_indirect' in f]

for file_path in woman_ind_files:
    print(file_path)
    analyze_effect_results(file_path, alt='woman')

for file_path in man_ind_files:
    print(file_path)
    analyze_effect_results(file_path, alt='man')
