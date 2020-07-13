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
    for f in files:
        print('Loading ' + f + '...')
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
        df['Example type'] = get_example_type(f)
        df['Effect type'] = 'Indirect' if 'indirect' in f else 'Direct'
        df['Yz'] = df['candidate2_prob'] / df['candidate1_prob']
        df['Y'] = df['candidate2_base_prob'] / df['candidate1_base_prob']
        df['Effect'] = df['Yz'] / df['Y'] - 1
        idx = df.groupby(['Layer', 'Neuron']).mean().sort_values('Effect')\
                .groupby('Layer').tail(int(len(df)*0.05)).index
        df['Top 5 percent'] = df.index.isin(idx)
        dfs.append(df)
    print('Concatentating...')
    df = pd.concat(dfs).reset_index()
    return df

def save_nie_by_layer_plot(df):
    print('Plotting nie by layer...')
    for et in EFFECT_TYPES:
        for ext in EXAMPLE_TYPES:
            try:
                data = df[(df['Effect type'] == et) 
                        & (df['Example type'] == ext)] 
                g = sns.FacetGrid(data=data,
                        row='Random', hue='Size', 
                        margin_titles=True, aspect=1.5)
                g.map(sns.lineplot, 'Layer', 'Effect')
                plt.tight_layout()
                plt.savefig(FIGURES_PATH + '_'.join(['nie',et,ext]) + '.svg')
                print('Success')
            except Exception as e: 
                print(e)

def draw_heatmap(data,color):
    pivot = data.groupby(['Layer','Neuron']).mean().reset_index()\
            .pivot(index='Layer', columns='Neuron',  values='Effect')
    ax = sns.heatmap(pivot, cbar=False, rasterized=True)
    ax.invert_yaxis()

def draw_heatmap_grid(df, et, r, ext):
    try:
        g = sns.FacetGrid(df, 
                row='Size',
                col='Example type', aspect=1.5, margin_titles=True)
        [[ax.title.set_position([.5, 1.5]) for ax in row] for row in  g.axes]
        g.map_dataframe(draw_heatmap)
        plt.tight_layout()
        plt.savefig(FIGURES_PATH + '_'.join(['heatmaps',r,et,ext]) + '.svg')
        print('Success')
    except Exception as e:
        print(e)

def save_heatmaps(df):
    print('Generating heatmaps...')
    for et in EFFECT_TYPES:
        for r in ['trained', 'random']:
            for ext in EXAMPLE_TYPES:
                f = ~df['Random'] if r == 'trained' else df['Random']
                data = df[(df['Effect type'] == et) 
                        & f & (df['Example type'] == ext)]
                draw_heatmap_grid(data, et, r, ext)

if __name__ == "__main__":
    df = load_dataframe_and_calculate_effects()
    save_nie_by_layer_plot(df)
    save_heatmaps(df)


