import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd

def main():

    models = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    model_to_name = {
        'distilgpt2': 'distill',
        'gpt2': 'small',
        'gpt2-medium': 'medium',
        'gpt2-large': 'large',
        'gpt2-xl': 'XL'
    }

    sns.set_context("paper")
    sns.set_style("white")

    palette = sns.color_palette()#('muted')

    filter = 'filtered'
    split = 'dev'
    dataset = 'winobias'

    te = []
    nde_all = []
    nie_all = []
    nie_sum = []
    model_names = []

    for model_version in models:
        fname = f"{dataset}_data/attention_intervention_{model_version}_{filter}_{split}.json"
        with open(fname) as f:
            data = json.load(f)
        df = pd.DataFrame(data['results'])
        # Convert to shape (num_examples X num_layers X num_heads)
        indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
        mean_sum_indirect_effect = indirect_by_head.sum(axis=(1, 2)).mean()
        te.append(data['mean_total_effect'])
        nde_all.append(data['mean_model_direct_effect'])
        nie_all.append(data['mean_model_indirect_effect'])
        nie_sum.append(mean_sum_indirect_effect)
        model_names.append(model_to_name[model_version])

    # Plot stacked bar chart
    plt.figure(num=1, figsize=(5, 3))
    width = 0.3
    inds = np.arange(len(models))
    p1 = plt.bar(inds, te, width, color=palette[2])
    p2 = plt.bar(inds + width, nie_all, width, color=palette[1])
    p3 = plt.bar(inds + width, nde_all, width, bottom=nie_all, color=palette[0])
    p4 = plt.bar(inds + 2 * width, nie_sum, width, color=palette[3])

    plt.ylabel('Effect', size=11)
    # plt.title('Effects of top heads', fontsize=11)
    plt.xticks(inds + .3, model_names, size=10)
    for tick in plt.gca().xaxis.get_minor_ticks():
        tick.label1.set_horizontalalignment('center')
    plt.yticks(size=10)
    # plt.yticks(np.arange(0, 81, 10))
    # p3 = plt.axhline(data['mean_total_effect'], linestyle='--')
    plt.legend((p1[0], p3[0], p2[0], p4[0]), ('TE', 'NDE-all', 'NIE-all', 'NIE-sum'), loc='upper left', fontsize=11)
    plt.savefig(f'results/attention_intervention/effects.pdf', format='pdf')
    plt.close()

if __name__ == '__main__':
    main()