import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import seaborn as sns

sns.set_context('talk')
sns.set_style('whitegrid')

PATH = sys.argv[1]
FIGURES_PATH = sys.argv[2]
by_feather = sys.argv[3].lower() == 'true'
MODELS = ['Distil', 'Small', 'Medium', 'Large', 'XL']
CHUNKSIZE = 100000
EFFECT_TYPES = ['Indirect', 'Direct']
EXAMPLE_TYPES = ['None', 'Distractor', 'Plural attractor', 
        'Singular attractor']
COLS = ['Layer', 'Neuron', 'Random', 'Model size', 'Intervening tokens', 
        'Effect type']
FORMAT = '.pdf'

def get_size(f):
    for m in MODELS:
        if m.lower() in f:
            return m
    return 'Small'

def get_example_type(f):
    for et in EXAMPLE_TYPES:
        if et.lower().split()[0] in f:
            return et

def load_dataframe_and_calculate_effects(by_feather=False):
    files = glob(PATH + '*.csv')
    preloaded = glob(PATH + '*.feather')
    dfs = []
    if by_feather:
        dfs = [pd.read_feather(f) for f in preloaded]
    else:
        for f in tqdm(files, desc='Loading files', leave=False):
            df = None
            feather = f.replace('csv', 'feather')
            if feather in preloaded:
                df = pd.read_feather(feather)
            else:
                df = pd.concat(tqdm(pd.read_csv(f, chunksize=CHUNKSIZE),
                    leave=False, desc='Loading dataframe for ' + f))
                df.to_feather(feather)
            df['Layer'] = df.layer
            df['Neuron'] = df.neuron
            df['Random'] = 'random' in f
            df['Model size'] = get_size(f)
            df['Intervening tokens'] = get_example_type(f)
            df['Effect type'] = 'Indirect' if 'indirect' in f else 'Direct'
            df['Yz'] = df.candidate2_prob / df.candidate1_prob
            df['Singular grammaticality'] = df.candidate2_base_prob \
                    / df.candidate1_base_prob
            df['Effect'] = df['Yz'] / df['Singular grammaticality'] - 1
            df['Plural grammaticality'] = df.candidate1_alt1_prob \
                    / df.candidate2_alt1_prob
            df['Total effect'] = 1 \
                    / (df['Plural grammaticality'] 
                            * df['Singular grammaticality']) \
                    - 1
            neurons = ['Neuron', 'Layer']
            df = df.set_index(neurons)
            neurons_per_layer = len(df.groupby('Neuron').mean().index)
            idx = df.groupby(neurons).mean().sort_values('Effect')\
                    .groupby('Layer')\
                    .tail(int(neurons_per_layer*0.05)).index
            df['Top 5 percent'] = df.index.isin(idx)
            dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df

def save_nie_by_layer_plot(df):
    print('Plotting nie by layer...')
    try:
        data = df[(df['Effect type'] == 'Indirect') & df['Top 5 percent']]\
                .groupby(COLS).mean().reset_index()
        g = sns.FacetGrid(data=data,
                col='Random', col_order=[False, True], 
                row='Intervening tokens', row_order=EXAMPLE_TYPES, 
                hue='Model size', hue_order=MODELS,
                height=5, aspect=2, 
                sharey=False)\
                        .map(sns.lineplot, 'Layer', 'Effect')
        [ax.legend() for ax in g.axes.flatten()]
        title = f'Indirect effects of top 5 percent of neurons by layer'
        plt.gcf().suptitle(title)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(FIGURES_PATH + title.lower().replace(' ', '_') + FORMAT)
        print('Success')
    except Exception as e: 
        print(e)

def draw_heatmap(data,color):
    pivot = data.groupby(['Layer', 'Neuron']).mean().reset_index()\
            .pivot(index='Layer', columns='Neuron', values='Effect')
    ax = sns.heatmap(pivot, rasterized=True)
    ax.invert_yaxis()    

def save_heatmaps(df):
    print('Generating heatmaps...')
    for et in EFFECT_TYPES:
        for r in ['trained', 'random']:
            f = ~df['Random'] if r == 'trained' else df['Random']
            data = df[(df['Effect type'] == et) & f]
            try:
                sns.FacetGrid(data, 
                        col='Model size', col_order=MODELS,
                        row='Intervening tokens', row_order=EXAMPLE_TYPES,
                        margin_titles=False,
                        aspect=2, height=5, 
                        sharey=False, sharex=False)\
                                .map_dataframe(draw_heatmap)
                title = f'{r.capitalize()} model {et.lower()} effect heatmaps'
                plt.suptitle(title)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(FIGURES_PATH 
                        + title.lower().replace(' ', '_') 
                        + FORMAT)
                print('Success')
            except Exception as e:
                print(e)

def save_aggregate_total_effect_bar(df):
    data = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            .groupby([c for c in COLS if c not in ['Layer', 'Neuron']]
                    + ['base_string', 'candidate1'])\
            .mean().reset_index()
    sns.FacetGrid(data, 
            row='Intervening tokens', row_order=EXAMPLE_TYPES,
            height=5, aspect=2,
            sharey=True, sharex=False)\
                    .map(sns.barplot, 'Model size', 'Total effect', 
                            orient='v', order=MODELS)\
                    .set(yscale='log')
    title = 'Total effects'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)

def save_y_comparisons(df):
    data = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            .groupby(['Model size', 'Intervening tokens'])\
            .mean().reset_index()
    sns.relplot(x='Singular grammaticality', y='Plural grammaticality',
            hue='Intervening tokens', hue_order=EXAMPLE_TYPES,
            size='Model size', size_order=reversed(MODELS),
            data=data)
    title = 'Model grammaticality'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.60, 0.95])
    plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)


if __name__ == "__main__":
    df = load_dataframe_and_calculate_effects(by_feather=by_feather)
    save_nie_by_layer_plot(df)
    save_heatmaps(df)
    save_aggregate_total_effect_bar(df)
    save_y_comparisons(df)

