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
MODELS = ['Distil', 'Small', 'Medium', 'Large', 'XL']
CHUNKSIZE = 100000
EFFECT_TYPES = ['Indirect', 'Direct']
EXAMPLE_TYPES = ['None', 'Distractor', 'Plural attractor', 
        'Singular attractor']

def get_size(f):
    for m in MODELS:
        if m.lower() in f:
            return m
    return 'Small'

def get_example_type(f):
    for et in EXAMPLE_TYPES:
        if et.lower().split()[0] in f:
            return et

def load_dataframe_and_calculate_effects():
    files = glob(PATH + '*.csv')
    preloaded = glob(PATH + '*.feather')
    dfs = []
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
        df = df.set_index(['Layer','Neuron'])
        df['Random'] = 'random' in f
        df['Size'] = get_size(f)
        df['Example'] = get_example_type(f)
        df['Effect type'] = 'Indirect' if 'indirect' in f else 'Direct'
        df['Yz'] = df['candidate2_prob'] / df['candidate1_prob']
        df['Y'] = df['candidate2_base_prob'] / df['candidate1_base_prob']
        df['Effect'] = df['Yz'] / df['Y'] - 1
        neurons_per_layer = len(df.groupby('Neuron').mean().index)
        idx = df.groupby(['Layer', 'Neuron']).mean().sort_values('Effect')\
                .groupby('Layer').tail(int(neurons_per_layer*0.05)).index
        df['Top 5 percent'] = df.index.isin(idx)
        dfs.append(df)
    df = pd.concat(dfs).reset_index()
    return df

def save_nie_by_layer_plot(df):
    print('Plotting nie by layer...')
    try:
        data = df[(df['Effect type'] == 'Indirect') & df['Top 5 percent']] 
        g = sns.FacetGrid(data=data,
                col='Random', row='Example', hue='Size',
                col_order=[False,True], 
                row_order=EXAMPLE_TYPES, 
                hue_order=MODELS,
                margin_titles=True, 
                height=4, 
                aspect=2, 
                sharey=False)
        g.map(sns.lineplot, 'Layer', 'Effect')
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + '_'.join(['nie']) + '.svg')
        print('Success')
    except Exception as e: 
        print(e)

def draw_heatmap(data,color):
    pivot = data.groupby(['Layer','Neuron']).mean().reset_index()\
            .pivot(index='Layer', columns='Neuron',  values='Effect')
    ax = sns.heatmap(pivot, rasterized=True)
    ax.invert_yaxis()    

def save_heatmaps(df):
    print('Generating heatmaps...')
    for et in EFFECT_TYPES:
        for r in ['trained', 'random']:
            f = ~df['Random'] if r == 'trained' else df['Random']
            data = df[(df['Effect type'] == et) & f]
            try:
                g = sns.FacetGrid(data, 
                        col='Size',
                        col_order=MODELS,
                        row='Example', 
                        row_order=EXAMPLE_TYPES,
                        margin_titles=False,
                        aspect=2, 
                        height=5, sharey=False, sharex=False)
                g.map_dataframe(draw_heatmap)
                plt.tight_layout()
                plt.savefig(FIGURES_PATH + '_'.join(['heatmaps',r,et]) + '.svg')
                print('Success')
            except Exception as e:
                print(e)

if __name__ == "__main__":
    df = load_dataframe_and_calculate_effects()
    save_nie_by_layer_plot(df)
    save_heatmaps(df)


