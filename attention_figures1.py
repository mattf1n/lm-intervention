"""
Creates figures for attention intervention analysis from JSON files:
    - Stacked bar chart with direct/indirect/total effects
    - Heatmap for head-level effects
    - Barplot for layer-level effects
    - Combined heatmap/barplot for head- and layer-level effects
"""

import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

from attention_utils import topk_indices

structure_to_title = {'simple': 'simple agreement',
                      'distractor': 'distractor',
                      'distractor_1': 'distractor (1)',
                      'singular': 'pp (singular)',
                      'plural': 'pp (plural)',
                      'pp': 'pp',
                      'rc_singular': 'across relative clause (singular)',
                      'rc_plural': 'across relative clause (plural)',
                      'rc': 'across relative clause',
                      'within_rc_singular': 'within relative clause (singular)',
                      'within_rc_plural': 'within relative clause (plural)',
                      'within_rc': 'within relative clause'}

def save_figures(data, source, model_version, filter, suffix=None, k=10):
    # Load data from json obj
    if source in ('rc', 'within_rc', 'pp'):
        results = data[0]['results']
        results.extend(data[1]['results'])
    else:
        results = data['results']
    df = pd.DataFrame(results)

    # Aggregate by head
    # Convert column to 3d ndarray (num_examples x num_layers x num_heads)
    indirect_by_head = np.stack(df['indirect_effect_head'].to_numpy())
    direct_by_head = np.stack(df['direct_effect_head'].to_numpy())
    # Average by head
    mean_indirect_by_head = indirect_by_head.mean(axis=0)
    std_indirect_by_head = indirect_by_head.std(axis=0)
    mean_direct_by_head = direct_by_head.mean(axis=0)
    std_direct_by_head = direct_by_head.std(axis=0)
    # Select top k heads by indirect effect
    topk_inds = topk_indices(mean_indirect_by_head, k)

    # Aggregate by layer
    # Convert column to 2d ndarray (num_examples x num_layers)
    indirect_by_layer = np.stack(df['indirect_effect_layer'].to_numpy())
    direct_by_layer = np.stack(df['direct_effect_layer'].to_numpy())
    mean_indirect_by_layer = indirect_by_layer.mean(axis=0)
    mean_direct_by_layer = direct_by_layer.mean(axis=0)
    n_layers = indirect_by_layer.shape[1]

    plt.rc('figure', titlesize=20)

    # Plot stacked bar chart
    palette = sns.color_palette()#('muted')
    plt.figure(num=1, figsize=(5, 2))
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
    if source in ("rc", "within_rc", "pp"):
        p3 = plt.axhline(data[0]['mean_total_effect'] + data[1]['mean_total_effect'] / 2, linestyle='--')
    else:
        p3 = plt.axhline(data['mean_total_effect'], linestyle='--')
    plt.legend((p3, p2[0], p1[0]), ('Total', 'Direct', 'Indirect'), loc='upper right', fontsize=11,
               bbox_to_anchor=(.99, 0.90))
    sns.despine()
    path = 'attention_figures/stacked_bar_charts'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{source}_{model_version}_{filter}.pdf', format='pdf')
    plt.close()
    annot = False

    # Plot heatmap for direct and indirect effect
    for effect_type in ('indirect', 'direct'):
        if effect_type == 'indirect':
            #mean_effect = mean_indirect_by_head
            mean_effect = std_indirect_by_head
        else:
            #mean_effect = mean_direct_by_head
            mean_effect = std_direct_by_head
        ax = sns.heatmap(mean_effect, rasterized=True, annot=annot, annot_kws={"size": 9}, fmt=".2f", square=True)
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect_type.capitalize()} Effect')
        plt.figure(num=1, figsize=(7, 5))
        path = f'attention_figures/heat_maps_std_{effect_type}'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/{source}_{model_version}_{filter}.pdf', format='pdf')
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
        path = f'attention_figures/layer_{effect_type}'
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/{source}_{model_version}_{filter}.pdf', format='pdf')
        plt.close()

    # Plot combined heatmap and barchart for direct and indirect effects
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
            fig = plt.figure(figsize=(3, 2.2))
            if model_version == 'distilgpt2':
                ax1 = plt.subplot2grid((100, 85), (0, 0), colspan=62, rowspan=99)
                ax2 = plt.subplot2grid((100, 85), (32, 69), colspan=17, rowspan=35)
            elif model_version in ('gpt2', 'gpt2_random'):
                ax1 = plt.subplot2grid((100, 85), (0, 0), colspan=65, rowspan=99)
                ax2 = plt.subplot2grid((100, 85), (12, 70), colspan=15, rowspan=75)
            elif model_version == 'gpt2-medium':
                ax1 = plt.subplot2grid((100, 85), (0, 5), colspan=55, rowspan=99)
                ax2 = plt.subplot2grid((100, 85), (2, 64), colspan=17, rowspan=95)
            elif model_version == 'gpt2-large':
                ax1 = plt.subplot2grid((100, 85), (0, 5), colspan=55, rowspan=96)
                ax2 = plt.subplot2grid((100, 85), (0, 62), colspan=17, rowspan=97)
            elif model_version == 'gpt2-xl':
                ax1 = plt.subplot2grid((100, 85), (0, 5), colspan=55, rowspan=96)
                ax2 = plt.subplot2grid((100, 85), (0, 62), colspan=17, rowspan=97)
            heatmap = sns.heatmap(effect_head, center=0.0, ax=ax1, annot=annot, annot_kws={"size": 9}, fmt=".2f", square=True, cbar=False, linewidth=0.1, linecolor='#D0D0D0',
            cmap = LinearSegmentedColormap.from_list('rg', ["#F14100", "white", "#3D4FC4"], N=256))
            plt.setp(heatmap.get_yticklabels(), fontsize=7)
            plt.setp(heatmap.get_xticklabels(), fontsize=7)
            heatmap.tick_params(axis='x', pad=1, length=2)
            heatmap.tick_params(axis='y', pad=1, length=2)
            heatmap.yaxis.labelpad = 2
            heatmap.invert_yaxis()
            if model_version != 'gpt2-xl':
                for i, label in enumerate(heatmap.xaxis.get_ticklabels()):
                    if i%2 == 1:
                        label.set_visible(False)
                for i, label in enumerate(heatmap.yaxis.get_ticklabels()):
                    if i%2 == 1:
                        label.set_visible(False)
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
            if model_version in ('gpt2-large', 'gpt2-xl'):
                cax = ax_divider.append_axes('left', size='7%', pad='45%')
            else:
                cax = ax_divider.append_axes('left', size='7%', pad='33%')
            # # make colorbar for heatmap.
            # # Heatmap returns an axes obj but you need to get a mappable obj (get_children)
            cbar = colorbar(ax1.get_children()[0], cax=cax, orientation='vertical')
            cax.yaxis.set_ticks_position('left')
            cbar.solids.set_edgecolor("face")
            cbar.ax.tick_params(labelsize=7, length=4, pad=2)
            ax1.set_title(structure_to_title[source], size=6)
            ax1.set_xlabel('Head', size=6)
            ax1.set_ylabel('Layer', size=6)
            for _, spine in ax1.spines.items():
                spine.set_visible(True)
            ax2.set_title('         Layer Effect', size=6)
            bp = sns.barplot(x=effect_layer, ax=ax2, y=list(range(n_layers)), color="#3D4FC4", orient="h")
            plt.setp(bp.get_xticklabels(), fontsize=7)
            bp.tick_params(axis='x', pad=1, length=3)
            ax2.invert_yaxis()
            ax2.set_yticklabels([])
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax2.xaxis.set_ticks_position('bottom')
            ax2.axvline(0, linewidth=.85, color='black')
            path = f'attention_figures/heat_maps_with_bar_{effect_type}{"_sorted" if do_sort else ""}'
            if not os.path.exists(path):
                os.makedirs(path)
            fname = f'{path}/{source}_{model_version}_{filter}.pdf'
            plt.savefig(fname, format='pdf', bbox_inches='tight')
            plt.close()

def main():
    sns.set_context("paper")
    sns.set_style("white")

    #model_versions = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    model_versions = ['gpt2']
    filters = ['unfiltered']
    #filters = ['filtered']
    structures = ['distractor', 'distractor_1', 'singular', 'plural', 'rc_singular', 'rc_plural', \
          'within_rc_singular', 'within_rc_plural', 'simple']

    # process structural bias
    for model_version in model_versions:
        for filter in filters:
            for structure in structures:
                fname = f"attention_results/{structure}/attention_intervention_{model_version}_{filter}.json"
                if not os.path.exists(fname):
                    print("File does not exist:", fname)
                    continue
                with open(fname) as f:
                    if structure in ("rc", "within_rc", "pp"):
                        file_str = f.readline()
                        json_strs = file_str.split("]},")
                        json_strs[0] += "]}"
                        data = [json.loads(json_str) for json_str in json_strs]
                    else:
                        data = json.load(f)
                    save_figures(data, structure, model_version, filter)
            

if __name__ == '__main__':
    main()
