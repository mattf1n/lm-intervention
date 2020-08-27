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

def save_nie_by_layer_plot():
    print('Plotting natural indirect effect of to 5% of neurons by layer...')
    df = pd.read_feather(PATH + 'agg.feather')
    data = df[(df['Effect type'] == 'Indirect') & df['Top 5 percent']]\
            .reset_index()
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
    plt.show()

if __name__ == "__main__":
    save_nie_by_layer_plot()

