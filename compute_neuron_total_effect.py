import numpy as np
import os
import pandas as pd
import sys


def compute_total_effect(row):
    if row['base_c1_effect'] >= 1.:
        return row['alt1_effect'] / row['base_c1_effect']
    else:
        return row['alt2_effect'] / row['base_c2_effect']


def main(folder_name="results/20191114_neuron_intervention/",
         model_name="distilgpt2"):
    fnames = [f[:-4] for f in os.listdir(folder_name) if f.endswith("csv")]
    fnames = [f for f in fnames if "_" + model_name in f]
    paths = [os.path.join(folder_name, f + ".csv") for f in fnames]
    # fnames[:5], paths[:5]
    woman_files = [f for f in paths
                   if "woman_indirect" in f
                   if os.path.exists(f.replace("indirect", "direct"))]

    means = []
    he_means = []
    she_means = []
    for path in woman_files:
        temp_df = pd.read_csv(path).groupby('base_string').agg('mean').reset_index()
        temp_df['alt1_effect'] = temp_df['candidate1_alt1_prob'] / temp_df['candidate2_alt1_prob']
        temp_df['alt2_effect'] = temp_df['candidate2_alt2_prob'] / temp_df['candidate1_alt2_prob']
        temp_df['base_c1_effect'] = temp_df['candidate1_base_prob'] / temp_df['candidate2_base_prob']
        temp_df['base_c2_effect'] = temp_df['candidate2_base_prob'] / temp_df['candidate1_base_prob']
        temp_df['he_total_effect'] = temp_df['alt1_effect'] / temp_df['base_c1_effect']
        temp_df['she_total_effect'] = temp_df['alt2_effect'] / temp_df['base_c2_effect']
        temp_df['total_effect'] = temp_df.apply(compute_total_effect, axis=1)

        mean_he_total = temp_df['he_total_effect'].mean()
        mean_she_total = temp_df['she_total_effect'].mean()
        mean_total = temp_df['total_effect'].mean()
        he_means.append(mean_he_total)
        she_means.append(mean_she_total)
        means.append(mean_total)

    print("The total effect of this model is {:.3f}".format(np.mean(means)))
    print("The total (male) effect of this model is {:.3f}".format(np.mean(he_means)))
    print("The total (female) effect of this model is {:.3f}".format(np.mean(she_means)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE: python ", sys.argv[0], "<model> <device> <out_dir>")
    # e.g., results/20191114...
    folder_name = sys.argv[1]
    # gpt2, gpt2-medium, gpt2-large
    model_name = sys.argv[2]

    main(folder_name, model_name)
