import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import os
import json
import winobias
import math
import csv

def winobias_figure():

    sns.set_context("paper")
    sns.set_style("white")
    model_version = 'gpt2-xl'
    print(model_version.upper())
    split = 'dev'
    filter = 'filtered'
    fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
    with open(fname) as f:
        data = json.load(f)

    x = []
    y = []
    for result in data['results']:
        # Get bias ratio
        female_pcts = []
        occupations = []
        for occupation, female_pct in winobias.OCCUPATION_FEMALE_PCT.items():
            if occupation in result['base_string1'].lower():
                female_pcts.append(female_pct)
                occupations.append((occupation, female_pct))
        assert len(female_pcts) == 2
        female_pct_ratio =  max(female_pcts) / min(female_pcts)
        print(occupations, female_pct_ratio, result['total_effect'])
        x.append(math.log(female_pct_ratio))
        y.append(math.log(result['total_effect']))

    plt.figure(figsize=(10,3))
    # ax = sns.lineplot(x,
    #              y,
    #              markers=True,
    #              dashes=True)
    ax = sns.scatterplot(x,
                 y)
    yticks = [-3, -2, -1, 0, 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$e^{{{y}}}$" for y in yticks])
    sns.despine()
    plt.show()

def winogender_figure():

    bergsma_pct_female = {}
    bls_pct_female = {}
    with open('winogender_data/winogender_occupation_stats.tsv') as f:
        next(f, None)  # skip the headers
        for row in csv.reader(f, delimiter='\t'):
            occupation = row[0]
            bergsma_pct_female[occupation] = float(row[1])
            bls_pct_female[occupation] = float(row[2])

    sns.set_context("paper")
    sns.set_style("white")

    model_version = 'gpt2'
    stats = 'bergsma'
    filter = 'filtered'
    fname = f"winogender_data/attention_intervention_{stats}_{model_version}_{filter}.json"
    with open(fname) as f:
        data = json.load(f)

    x = []
    y = []
    for result in data['results']:
        # Get bias ratio
        female_pcts = []
        occupations = []
        for occupation, female_pct in bergsma_pct_female.items():
            if occupation in result['base_string1'].lower():
                female_pcts.append(female_pct)
                occupations.append(occupation)
        assert len(female_pcts) == 1
        female_pcts.append(0.5) # Assume the participant is 50% likely to be female
        female_pct_ratio =  max(female_pcts) / min(female_pcts)
        print(occupations, female_pct, result['total_effect'])
        x.append(math.log(female_pct_ratio))
        y.append(math.log(result['total_effect']))

    plt.figure(figsize=(10,3))
    ax = sns.scatterplot(x,
                 y)#,
                 # markers=True,
                 # dashes=True)
    yticks = [-3, -2, -1, 0, 1]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$e^{{{y}}}$" for y in yticks])
    sns.despine()
    plt.show()


#
# def get_profession_correlation(total_df, direction="woman"):
#     x_vals = []
#     y_vals = []
#     labels = []
#     total_by_ex = total_df.groupby('base_string_direct').agg('mean')
#     for index, row in total_by_ex.iterrows():
#         if abs(row['total_causal_effect']) > 100:
#             continue
#         labels.append(index.split()[1])
#         y_vals.append(row['total_causal_effect'])
#         x_vals.append(profession_stereotypicality[index.split()[1]]['max'])
#     profession_df = pd.DataFrame({'example': labels,
#                                   'bias': x_vals,
#                                   'log-odds': np.log(y_vals)})
#

winobias_figure()