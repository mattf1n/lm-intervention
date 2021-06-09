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

def save_aggregate_total_effect_bar():
    try: df = pd.read_feather(PATH + 'effects.feather')
    except: 
        print(PATH + 'effects.feather not found.' 
                + f'Run `make_feathers.py {PATH}` to generate.')
        return
    data = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            .groupby(['Model size', 'Intervening tokens', 'base_string', 
                'candidate1'])\
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
    plt.show()

if __name__ == "__main__":
    save_aggregate_total_effect_bar()

