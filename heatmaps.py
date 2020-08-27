import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

sns.set()

PATH = sys.argv[1]
FIGURES_PATH = sys.argv[2]
MODELS = ['Distil', 'Small', 'Medium', 'Large', 'XL']
EXAMPLE_TYPES = ['None', 'Distractor', 'Plural attractor', 
        'Singular attractor']
FORMAT = '.pdf'

def draw_heatmap(data,color):
    pivot = data.pivot(index='Layer', columns='Neuron', values='Effect')
    ax = sns.heatmap(pivot, rasterized=True)
    ax.invert_yaxis()    

def save_heatmaps():
    print('Generating heatmaps...')
    df = pd.read_feather(PATH + 'agg.feather')
    data = df[(df['Effect type'] == 'Indirect') & ~df['Random']]
    sns.FacetGrid(data, 
            col='Model size', col_order=MODELS,
            row='Intervening tokens', row_order=EXAMPLE_TYPES,
            margin_titles=False,
            aspect=2, height=5, 
            sharey=False, sharex=False).map_dataframe(draw_heatmap)
    title = f'Indirect effect heatmaps'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(FIGURES_PATH 
            + title.lower().replace(' ', '_') 
            + FORMAT)
    plt.show()

if __name__ == "__main__":
    save_heatmaps()

