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

def save_y_comparisons():
    df = pd.read_feather(PATH + 'effects.feather')
    data0 = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            .groupby(['Model size', 'Intervening tokens', 'base_string', 
                'candidate1'])\
            .mean().reset_index()
    data1 = data0.copy()
    data0['Grammaticality'] = 1 / data0['Plural grammaticality']
    data1['Grammaticality'] = 1 / data0['Singular grammaticality']
    data0['Agreement'] = 'Plural'
    data1['Agreement'] = 'Singular'
    data = pd.concat([data0, data1])
    sns.catplot(y='Grammaticality', x='Intervening tokens',
            col='Agreement', 
            hue='Model size', hue_order=reversed(MODELS),
            data=data, sharey=True, kind='point', dodge=True, aspect=1.5)
                .set(yscale='log')
    title = 'Model grammaticality'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.95, 0.90])
    plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)
    # plt.show()

if __name__ == "__main__":
    save_y_comparisons()

