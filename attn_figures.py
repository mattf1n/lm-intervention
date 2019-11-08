import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from attention_utils import topk_indices
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


def save_figures(data, source, model_version, filter, suffix, k=10):

    # Load data from json obj
    results = data['results']
    df = pd.DataFrame(results)

    # Aggregate by head
    # Convert column to 3d ndarray (num_examples x num_layers x num_heads)
    indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
    direct_by_head = np.stack(df['direct_effect_head'].to_numpy())
    # Average by head
    mean_indirect_by_head = indirect_by_head.mean(axis=0)
    mean_direct_by_head = direct_by_head.mean(axis=0)
    # Select top k heads by indirect effect
    topk_inds = topk_indices(mean_indirect_by_head, k)

    #Aggregate by layer
    # Convert column to 2d ndarray (num_examples x num_layers)
    indirect_by_layer = np.stack(df['indirect_effect_layer'].to_numpy())
    direct_by_layer = np.stack(df['direct_effect_layer'].to_numpy())
    mean_indirect_by_layer = indirect_by_layer.mean(axis=0)
    mean_direct_by_layer = direct_by_layer.mean(axis=0)
    n_layers = indirect_by_layer.shape[1]

    plt.rc('figure', titlesize=20)

    # Plot stacked bar chart
    topk_direct = []
    topk_indirect = []
    labels = []
    for ind in topk_inds:
        layer, head = np.unravel_index(ind, mean_indirect_by_head.shape)
        topk_indirect.append(mean_indirect_by_head[layer, head])
        topk_direct.append(mean_direct_by_head[layer, head])
        labels.append(f'{layer}-{head}')
    width = 0.6
    inds = range(k)
    p1 = plt.bar(inds, topk_direct, width)
    p2 = plt.bar(inds, topk_indirect, width, bottom=topk_direct)
    plt.ylabel('Effect')
    plt.title('Effects of top heads', fontsize=14)
    plt.xticks(inds, labels, size=12)
    plt.yticks(size=12)
    # plt.yticks(np.arange(0, 81, 10))
    p3 = plt.axhline(data['mean_total_effect'], linestyle='--')
    plt.legend((p3, p2[0], p1[0]), ('Total', 'Indirect', 'Direct'), loc='lower right', fontsize=14)
    plt.figure(num=1, figsize=(10, 15))
    plt.savefig(f'results/attention_intervention/stacked_bar_charts/{source}_{model_version}_{filter}_'
                f'{suffix}.png', format='png')
    plt.close()

    annot = False

    # Plot heatmap for direct and indirect effect
    for effect_type in ('indirect', 'direct'):
        if effect_type == 'indirect':
            mean_effect = mean_indirect_by_head
        else:
            mean_effect = mean_direct_by_head
        ax = sns.heatmap(mean_effect, annot=annot, annot_kws={"size": 9}, fmt=".2f", square=True)
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')
        plt.figure(num=1, figsize=(7, 5))
        plt.savefig(f'results/attention_intervention/heat_maps_{effect_type}/{source}_{model_version}_{filter}_'
                    f'{suffix}.png', format='png')
        plt.close()

    # Plot layer-level bar chart for indirect and direct effects
    for effect_type in ('indirect', 'direct'):
        if effect_type == 'indirect':
            mean_effect = mean_indirect_by_layer
        else:
            mean_effect = mean_direct_by_layer
        plt.figure(num=1, figsize=(5, 5))
        ax = sns.barplot(x=mean_effect, y=list(range(n_layers)), orient="h", color="#4472C4")
        ax.set(ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')
        # ax.axvline(0, linewidth=.85, color='black')
        plt.savefig(f'results/attention_intervention/layer_{effect_type}/{source}_{model_version}_{filter}_'
                    f'{suffix}.png', format='png')
        plt.close()

    # Plot heatmap + barplot for direct and indirect effect
    for effect_type in ('indirect', 'direct'):
        if effect_type == 'indirect':
            effect_head = mean_indirect_by_head
            effect_layer = mean_indirect_by_layer
        else:
            effect_head = mean_direct_by_head
            effect_layer = mean_direct_by_layer
        fig = plt.figure(figsize=(4.3, 3.8))
        ax1 = plt.subplot2grid((100, 85), (0, 0), colspan=60, rowspan=99)
        ax2 = plt.subplot2grid((100, 85), (0, 62), colspan=14, rowspan=75)
        sns.heatmap(effect_head, ax=ax1, annot=annot, annot_kws={"size": 9}, fmt=".2f", square=True, cbar=False)
        # split axes of heatmap to put colorbar
        ax_divider = make_axes_locatable(ax1)
        # # define size and padding of axes for colorbar
        cax = ax_divider.append_axes('bottom', size='7%', pad='25%')
        # # make colorbar for heatmap.
        # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
        colorbar(ax1.get_children()[0], cax=cax, orientation='horizontal')
        # # locate colorbar ticks
        cax.xaxis.set_ticks_position('bottom')
        ax1.set(xlabel='Head', ylabel='Layer', title='Head Effect')
        ax2.set(title=f'     Layer Effect')
        # sns.set_style("ticks")
        sns.barplot(x=effect_layer, ax=ax2, y=list(range(n_layers)), color="#4472C4", orient="h")
        # ax2.set_frame_on(False)
        ax2.set_yticklabels([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        # ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.xaxis.set_ticks_position('bottom')
        ax2.axvline(0, linewidth=.85, color='black')
        plt.figure(num=1, figsize=(14, 10))
        # plt.show()
        plt.savefig(f'results/attention_intervention/heat_maps_with_bar_{effect_type}/{source}_{model_version}_{filter}_'
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
                    save_figures(data, 'winobias', model_version, filter, split)

    # Process winogender
    for model_version in model_versions:
        for filter in filters:
            for stat in ['bergsma', 'bls']:
                fname = f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
                    save_figures(data, 'winogender', model_version, filter, stat)


if __name__ == '__main__':
    main()