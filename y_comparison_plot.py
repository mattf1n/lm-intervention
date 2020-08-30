import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns

sns.set()

PATH = sys.argv[1]
FIGURES_PATH = sys.argv[2]
AGG = sys.argv[3] == 'true'
MODELS = ['Distil', 'Small', 'Medium', 'Large', 'XL']
EXAMPLE_TYPES = ['None', 'Distractor', 'Plural attractor', 
        'Singular attractor']
FORMAT = '.pdf'

gb = ['Model size', 'Intervening tokens'] if AGG \
        else ['Model size', 'Intervening tokens', 'base_string', 'candidate1']

def save_y_comparisons():
    df = pd.read_feather(PATH)
    data = df[~df.Random & (df['Effect type'] == 'Indirect')]\
            .groupby(gb)\
            .mean().reset_index()\
            .melt(
                    id_vars=gb,
                    value_vars=['Singular grammaticality', 
                        'Plural grammaticality'],
                    var_name='Type', value_name='Grammaticality')
    data['Agreement'] = data.Type.str.split().str.get(0)
    data['Grammaticality'] = 1 / data['Grammaticality']
    sns.catplot(y='Grammaticality', x='Intervening tokens',
            col='Agreement', 
            hue='Model size', hue_order=reversed(MODELS),
            data=data, sharey=True, kind='point', dodge=True, aspect=1.5)\
                    .set(yscale='log')
    title = 'Model grammaticality'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.925, 0.90])
    plt.savefig(FIGURES_PATH + f'{title.lower().replace(" ", "_")}' + FORMAT)
    # plt.show()

if __name__ == "__main__":
    save_y_comparisons()

