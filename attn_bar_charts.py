import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from attention_utils import topk_indices


def save_bar_chart(data, source, model_version, filter, suffix, k=10):

    # Load data from json obj
    results = data['results']
    df = pd.DataFrame(results)
    # Convert column to 3d ndarray (num_examples x num_layers x num_heads)
    indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
    direct_by_head = np.stack(df['direct_effect_head'].to_numpy())
    # Average by head
    mean_indirect_by_head = indirect_by_head.mean(axis=0)
    mean_direct_by_head = direct_by_head.mean(axis=0)
    # Select top k heads by indirect effect
    topk_inds = topk_indices(mean_indirect_by_head, k)

    # Populate data for figure
    topk_direct = []
    topk_indirect = []
    labels = []
    for ind in topk_inds:
        layer, head = np.unravel_index(ind, mean_indirect_by_head.shape)
        topk_indirect.append(mean_indirect_by_head[layer, head])
        topk_direct.append(mean_direct_by_head[layer, head])
        labels.append(f'{layer}-{head}')

    # Plot and save figure
    width = 0.6
    inds = range(k)
    p1 = plt.bar(inds, topk_direct, width)
    p2 = plt.bar(inds, topk_indirect, width, bottom=topk_direct)
    plt.ylabel('Effect', fontsize=14)
    plt.title('Effects of top heads', fontsize=14)
    plt.xticks(inds, labels, size=12)
    plt.yticks(size=12)
    # plt.yticks(np.arange(0, 81, 10))
    p3 = plt.axhline(data['mean_total_effect'], linestyle='--')
    plt.legend((p3, p2[0], p1[0]), ('Total', 'Indirect', 'Direct'), loc='lower right', fontsize=14)
    plt.figure(num=1, figsize=(10, 15))
    plt.savefig(f'results/attention_intervention/bar_charts/{source}_{model_version}_{filter}_'
                f'{suffix}.png', format='png')
    plt.close()


def main():
    sns.set_context("paper")
    sns.set_style("white")

    model_versions = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large']
    filters = ['filtered', 'unfiltered']

    # Process winobias
    for model_version in model_versions:
        for filter in filters:
            for split in ['dev', 'test']:
                fname =  f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                with open(fname) as f:
                    data = json.load(f)
                    save_bar_chart(data, 'winobias', model_version, filter, split)

    # Process winogender
    for model_version in model_versions:
        for filter in filters:
            for stat in ['bergsma', 'bls']:
                fname = f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
                    save_bar_chart(data, 'winogender', model_version, filter, stat)


if __name__ == '__main__':
    main()