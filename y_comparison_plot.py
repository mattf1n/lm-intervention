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
    data0['Grammaticality'] = data0['Plural grammaticality']
    data1['Grammaticality'] = data0['Singular grammaticality']
    data0['Grammaticality type'] = 'Plural'
    data1['Grammaticality type'] = 'Singular'
    data = pd.concat([data0, data1])
    sns.catplot(x='Grammaticality', y='Intervening tokens',
            col='Grammaticality type', 
            hue='Model size', hue_order=reversed(MODELS),
            data=data, sharex=True)
    title = 'Model grammaticality'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.75, 0.95])
    plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)
    # plt.show()

# def save_y_comparisons():
    # df = pd.read_feather(PATH + 'agg.feather')
    # data = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            # .groupby(['Model size', 'Intervening tokens'])\
            # .mean().reset_index()
    # sns.relplot(x='Singular grammaticality', y='Plural grammaticality',
            # hue='Intervening tokens', hue_order=EXAMPLE_TYPES,
            # size='Model size', size_order=reversed(MODELS),
            # data=data).set(xlim=(0,0.6), ylim=(0,0.6))
    # title = 'Model grammaticality'
    # plt.suptitle(title)
    # plt.tight_layout(rect=[0, 0, 0.75, 0.95])
    # plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)
    # # plt.show()

if __name__ == "__main__":
    save_y_comparisons()

