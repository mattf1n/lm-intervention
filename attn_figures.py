import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from attention_utils import topk_indices
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import os

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
    palette = sns.color_palette()#('muted')
    plt.figure(num=1, figsize=(5, 3))
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
    p1 = plt.bar(inds, topk_indirect, width, linewidth=0, color=palette[1])
    p2 = plt.bar(inds, topk_direct, width, bottom=topk_indirect, linewidth=0, color=palette[0])
    plt.ylabel('Effect', size=11)
    plt.title('Effects of top heads', fontsize=11)
    plt.xticks(inds, labels, size=10)
    plt.yticks(size=10)
    # plt.yticks(np.arange(0, 81, 10))
    p3 = plt.axhline(data['mean_total_effect'], linestyle='--')
    plt.legend((p3, p2[0], p1[0]), ('Total', 'Direct', 'Indirect'), loc='lower right', fontsize=11)
    plt.savefig(f'results/attention_intervention/stacked_bar_charts/{source}_{model_version}_{filter}_'
                f'{suffix}.pdf', format='pdf')
    plt.close()

    annot = False

    # Plot heatmap for direct and indirect effect
    for effect_type in ('indirect', 'direct'):
        if effect_type == 'indirect':
            mean_effect = mean_indirect_by_head
        else:
            mean_effect = mean_direct_by_head
        ax = sns.heatmap(mean_effect, rasterized=True, annot=annot, annot_kws={"size": 9}, fmt=".2f", square=True)
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')
        plt.figure(num=1, figsize=(7, 5))
        plt.savefig(f'results/attention_intervention/heat_maps_{effect_type}/{source}_{model_version}_{filter}_'
                    f'{suffix}.pdf', format='pdf')
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
                    f'{suffix}.pdf', format='pdf')
        plt.close()

    for do_sort in False, True:
        for effect_type in ('indirect', 'direct'):
            if effect_type == 'indirect':
                effect_head = mean_indirect_by_head
                effect_layer = mean_indirect_by_layer
                if do_sort:
                    effect_head = -np.sort(-effect_head) # Sort indirect effects within each layer in descending order
            else:
                if do_sort:
                    continue
                effect_head = mean_direct_by_head
                effect_layer = mean_direct_by_layer
            fig = plt.figure(figsize=(4.3, 3.8))
            if model_version == 'distilgpt2':
                ax1 = plt.subplot2grid((100, 85), (0, 2), colspan=60, rowspan=99)
                ax2 = plt.subplot2grid((100, 85), (17, 67), colspan=17, rowspan=41)
            elif model_version in ('gpt2', 'gpt2_random'):
                ax1 = plt.subplot2grid((100, 85), (0, 0), colspan=60, rowspan=99)
                ax2 = plt.subplot2grid((100, 85), (0, 62), colspan=15, rowspan=75)
            elif model_version == 'gpt2-medium':
                ax1 = plt.subplot2grid((100, 85), (0, 12), colspan=40, rowspan=75)
                ax2 = plt.subplot2grid((100, 85), (0, 54), colspan=17, rowspan=75)
            elif model_version == 'gpt2-large':
                ax1 = plt.subplot2grid((100, 85), (0, 16), colspan=32, rowspan=75)
                ax2 = plt.subplot2grid((100, 85), (0, 51), colspan=17, rowspan=75)
            elif model_version == 'gpt2-xl':
                ax1 = plt.subplot2grid((100, 85), (0, 16), colspan=32, rowspan=75)
                ax2 = plt.subplot2grid((100, 85), (0, 51), colspan=17, rowspan=75)
            heatmap = sns.heatmap(effect_head, rasterized=True, ax=ax1, annot=annot, annot_kws={"size": 9}, fmt=".2f",
                        square=True, cbar=False)
            if do_sort:
                heatmap.axes.get_xaxis().set_ticks([])
            else:
                if model_version == 'gpt2-xl':
                    every_nth = 2
                    for n, label in enumerate(ax1.xaxis.get_ticklabels()):
                        if n % every_nth != 0:
                            label.set_visible(False)
                    for n, label in enumerate(ax1.yaxis.get_ticklabels()):
                        if n % every_nth != 0:
                            label.set_visible(False)
            # split axes of heatmap to put colorbar
            ax_divider = make_axes_locatable(ax1)
            # # define size and padding of axes for colorbar
            if model_version == 'distilgpt2':
                cax = ax_divider.append_axes('bottom', size='10%', pad='50%')
            elif model_version in ('gpt2-xl', 'gpt2-large', 'gpt2-medium'):
                cax = plt.subplot2grid((100, 85), (95, 10), colspan=45, rowspan=4)
            else:
                cax = ax_divider.append_axes('bottom', size='7%', pad='25%')
            # # make colorbar for heatmap.
            # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
            cbar = colorbar(ax1.get_children()[0], cax=cax, orientation='horizontal')
            cbar.solids.set_edgecolor("face")
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
            # plt.figure(num=1, figsize=(14, 10))
            # plt.show()
            fname = f'results/attention_intervention/heat_maps_with_bar_{effect_type}{"_sorted" if do_sort else ""}/'\
                    f'{source}_{model_version}_{filter}_{suffix}.pdf'
            plt.savefig(fname, format='pdf')
            plt.close()

def main():
    sns.set_context("paper")
    sns.set_style("white")

    model_versions = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt2_random']
    filters = ['filtered', 'unfiltered']

    # For testing:
    #
    # model_version = 'gpt2'
    # split = 'dev'
    # filter = 'filtered'
    # fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
    # with open(fname) as f:
    #     data = json.load(f)
    #     save_figures(data, 'winobias', model_version, filter, split)
    # return

    # Process winobias
    for model_version in model_versions:
        for filter in filters:
            for split in ['dev', 'test']:
                fname =  f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                if not os.path.exists(fname):
                    print('File does not exist:', fname)
                    continue
                with open(fname) as f:
                    data = json.load(f)
                    save_figures(data, 'winobias', model_version, filter, split)

    # Process winogender
    for model_version in model_versions:
        for filter in filters:
            for stat in ['bergsma', 'bls']:
                fname = f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                if not os.path.exists(fname):
                    print('File does not exist:', fname)
                    continue
                with open(fname) as f:
                    data = json.load(f)
                    save_figures(data, 'winogender', model_version, filter, stat)


if __name__ == '__main__':
    main()