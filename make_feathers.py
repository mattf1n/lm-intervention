import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import sys

PATH = sys.argv[1]
MODELS = ['Distil', 'Small', 'Medium', 'Large', 'XL']
CHUNKSIZE = 100000
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

def compute_effects_and_save():
    files = glob(PATH + '*.csv')
    preloaded = glob(PATH + '*.feather')
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
        with_effects = os.path.splitext(f)[0] + '_with_effects.feather'
        df.to_feather(with_effects)
        aggregate = os.path.split(f)[0] + '_agg.feather'
        df.to_feather(aggregate)
        dfs.append(df)

if __name__ == "__main__":
    compute_effects_and_save()

