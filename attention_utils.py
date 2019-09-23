import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm
import pandas as pd
import numpy as np


def perform_intervention(intervention, model, effects=('indirect', 'direct')):
    """Perform intervention and return results for specified effects"""
    x = intervention.base_strings_tok[0]  # E.g. The doctor asked the nurse a question. He
    x_alt = intervention.base_strings_tok[1]  # E.g. The doctor asked the nurse a question. She

    with torch.no_grad():
        candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples(
            x,
            intervention.candidates_tok)
        candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples(
            x_alt,
            intervention.candidates_tok)

    candidate1 = ' '.join(intervention.candidates[0]).replace('Ġ', '')
    candidate2 = ' '.join(intervention.candidates[1]).replace('Ġ', '')

    odds_base = candidate2_base_prob / candidate1_base_prob
    odds_alt = candidate2_alt_prob / candidate1_alt_prob
    odds_ratio_total = odds_alt / odds_base

    results = {
        'base_string1': intervention.base_strings[0],
        'base_string2': intervention.base_strings[1],
        'candidate1': candidate1,
        'candidate2': candidate2,
        'candidate1_base_prob': candidate1_base_prob,
        'candidate2_base_prob': candidate2_base_prob,
        'odds_base': odds_base,
        'candidate1_alt_prob': candidate1_alt_prob,
        'candidate2_alt_prob': candidate2_alt_prob,
        'odds_alt': odds_alt,
        'odds_ratio_total': odds_ratio_total,
    }
    odds_ratio_indirect = None
    odds_ratio_direct = None
    for effect in effects:
        candidate1_probs, candidate2_probs = model.attention_intervention_experiment(intervention, effect)
        odds_intervention = candidate2_probs / candidate1_probs
        odds_ratio_intervention = odds_intervention / odds_base
        if effect == 'indirect':
            odds_ratio_indirect = odds_ratio_intervention.tolist()
        else:
            odds_ratio_direct = odds_ratio_intervention.tolist()
    results['odds_ratio_indirect'] = odds_ratio_indirect
    results['odds_ratio_direct'] = odds_ratio_direct
    return results


def report_intervention(results, effects=('indirect', 'direct'), verbose=False):

    print(f"x : {results['base_string1']}")
    print(f"x': {results['base_string2']}")
    print(f"c1: {results['candidate1']}")
    print(f"c2: {results['candidate2']}")
    print(f"\np(c2|x) / p(c1|x) = {results['odds_base']:.5f}")
    print(f"p(c2|x') / p(c1|x') = {results['odds_alt']:.5f}")
    print(f"\nTOTAL Effect: (p(c2|x') / p(c1|x')) / (p(c2|x) / p(c1|x)) = {results['odds_ratio_total']:.3f}")

    for effect in effects:
        if verbose:
            print(f'\n{effect.upper()} Effect')
            if effect == 'indirect':
                print("   Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
                print(f"   Effect = (p(c2|x, Attn(x')) / p(c1|x, Attn(x')) / (p(c2|x) / p(c1|x))")
            elif effect == 'direct':
                print("   Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
                print(f"   Effect = (p(c2|x', Attn(x)) / p(c1|x', Attn(x)) / (p(c2|x) / p(c1|x))")
            else:
                raise ValueError(f"Invalid effect: {effect}")

        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(results['odds_ratio_' + effect], annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'{effect.capitalize()} Effect')


def perform_interventions(interventions, model, effects=('indirect', 'direct')):
    results_list = []
    for intervention in tqdm(interventions):
        results = perform_intervention(intervention, model, effects)
        results_list.append(results)
    return results_list


def report_interventions_summary(results, effects=('indirect', 'direct'), verbose=False):

    df = pd.DataFrame(results)

    print('*** SUMMARY ***')
    print(f"Num interventions: {len(df)}")
    print(f"Mean total effect: {df.odds_ratio_total.mean():.2f}")

    for effect in effects:
        odds_ratio_intervention = np.stack(df['odds_ratio_' + effect].to_numpy()) # Convert column to 3d ndarray
        mean_effect = odds_ratio_intervention.mean(axis=0)
        # mean_effect = odds_ratio_intervention.mean() # This doesn't work when you load from csv file

        if verbose:
            print(f'\n{effect.upper()} Effect')
            if effect == 'indirect':
                print("   Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
                print(f"   Effect = (p(c2|x, Attn(x')) / p(c1|x, Attn(x')) / (p(c2|x) / p(c1|x))")
            elif effect == 'direct':
                print("   Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
                print(f"   Effect = (p(c2|x', Attn(x)) / p(c1|x', Attn(x)) / (p(c2|x) / p(c1|x))")
        plt.figure(figsize=(9, 7))

        ax = sns.heatmap(mean_effect, annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect.capitalize()} Effect')

if __name__ == "__main__":
    from pytorch_transformers import GPT2Tokenizer
    from experiment import Intervention, Model
    from pandas import DataFrame
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(output_attentions=True)

    # Test experiment
    interventions = [
        Intervention(
            tokenizer,
            "The doctor asked the nurse a question. {}",
            ["He", "She"],
            ["asked", "answered"]),
        Intervention(
            tokenizer,
            "The doctor asked the nurse a question. {}",
            ["He", "She"],
            ["requested", "responded"])
    ]

    results = perform_interventions(interventions, model)
    report_interventions_summary(results)

    df = DataFrame(results)
    report_interventions_summary(df)



