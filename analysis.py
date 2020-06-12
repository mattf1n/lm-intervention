import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

PATH = sys.argv[1]
FIGURES_PATH = sys.argv[2]
MODELS = ['D', 'S', 'M', 'L', 'XL', 'Rand']

class experiment():
    def __init__(self, filename):
        self.filename = os.path.splitext(filename)[0].split('/')[-1]
        if 'distil' in self.filename:
            self.model = 'D'
        elif 'medium' in self.filename:
            self.model = 'M'
        elif 'large' in self.filename:
            self.model = 'L'
        elif 'xl' in self.filename:
            self.model = 'XL'
        elif 'rand' in self.filename:
            self.model = 'Rand'
        else:
            self.model = 'S'
        self.random = 'random' in self.filename
        self.get_df()
        self.get_effects()
        self.save_heatmap()

    def is_analyzed(self):
        return (PATH + self.filename + '.txt') in glob(PATH + '*')
    
    def is_loaded(self):
        return (PATH + self.filename + '.feather') in glob(PATH + '*')

    def get_df(self):
        if not self.is_loaded():
            CHUNKSIZE = 100000
            df_list = [df_chunk for df_chunk 
                    in tqdm(pd.read_csv(PATH + self.filename + '.csv', 
                        chunksize=CHUNKSIZE), leave=False, 
                        desc='Loading dataframe')]
            self.df = pd.concat(df_list)
            del df_list
            self.df.to_feather(PATH + self.filename + '.feather')
        else:
            self.df = pd.read_feather(PATH + self.filename + '.feather')
        self.number_of_layers = int(max(self.df['layer']))
        self.neurons_per_layer = int(max(self.df['neuron']))
        return self.df

    def get_total_effect(self):
        df = self.df
        iso = df.drop_duplicates(subset=['candidate1', 'word'])
        yx = iso['candidate2_alt1_prob'] / iso['candidate1_alt1_prob']
        y = iso['candidate2_base_prob'] / iso['candidate1_base_prob']
        total_effects = yx/y - 1
        self.te = total_effects.mean()
        return self.te

    def get_effects(self, attractor='all'):
        if self.is_analyzed():
            self.effects = np.loadtxt(PATH + self.filename + '.txt')
        else:
            df = self.df
            ies = np.zeros((self.number_of_layers, self.neurons_per_layer))
            for l in tqdm(range(self.number_of_layers), desc='Layers', 
                    leave=False):
                for n in tqdm(range(self.neurons_per_layer), leave=False,
                        desc='Neurons'):
                    iso = df[(df['layer'] == l) & (df['neuron'] == n)]
                    yz = iso['candidate2_prob'] / iso['candidate1_prob']
                    y = iso['candidate2_base_prob'] \
                            / iso['candidate1_base_prob']
                    ie = yz/y - 1
                    ies[l][n] = ie.mean()
            np.savetxt(PATH + self.filename + '.txt', ies)
            self.effects = ies
            return ies

    def get_top(self):
        self.top = np.array([max(layer) for layer in self.effects])
        pct = len(self.effects[0]) // 20
        self.top_5pct = np.array([np.mean(np.partition(layer,-pct)[-pct:]) 
            for layer in self.effects])

    def save_heatmap(self):
        plt.figure(dpi=1000)
        plt.imshow(self.effects, aspect='auto', cmap='inferno',
                interpolation=None, origin='lower')
        plt.ylabel('Layer')
        plt.xlabel('Neuron')
        plt.colorbar()
        plt.savefig(FIGURES_PATH + self.filename + '.pdf')
        plt.clf()

def save_nie_chart(experiments, top=True):
    postfix = 'top' if top else 'top5'
    for exp in tqdm(experiments, leave=False, desc='Saving NIE chart for ' + postfix):
        exp.get_top()
        color = 'blue'
        if 'random' in exp.filename:
            color = 'gray'
        elif 'plural' in exp.filename:
            color = 'red'
        elif 'singular' in exp.filename:
            color = 'green'
        if top:
            plt.plot(exp.top, color)
        else:
            plt.plot(exp.top_5pct, color)
        plt.ylabel('Natural Indirect Effect')
        plt.xlabel('Layer')
    plt.savefig(FIGURES_PATH + 'nie.pdf')
    plt.clf()

def save_ate_chart(experiments):
    plt.figure(figsize=(4,3))
    iso = [exp for exp in experiments]
    iso.sort(key=lambda exp: MODELS.index(exp.model))
    titles = [exp.model for exp in iso]
    total_effects = [exp.get_total_effect() 
            for exp in tqdm(iso, leave=False, desc='Saving ATE chart')]
    plt.bar(titles, total_effects)
    plt.ylabel('Total Effect')
    plt.xlabel('Model')
    plt.savefig(FIGURES_PATH + 'ate.pdf')
    plt.clf()

if __name__ == "__main__":
    experiments = [experiment(filename) 
        for filename in tqdm(glob(PATH + '*indirect*.csv'), leave=False, 
            desc='Loading experiments')]

    save_nie_chart(experiments)
    save_nie_chart(experiments, top=False)
    save_ate_chart(experiments)
