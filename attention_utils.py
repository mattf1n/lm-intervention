import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def report_intervention(intervention, model, effects=('indirect', 'direct'), verbose=False):

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
    odds_ratio_alt = odds_alt / odds_base

    print(f"x : {intervention.base_strings[0]}")
    print(f"x': {intervention.base_strings[1]}")
    print(f"c1: {candidate1}")
    print(f"c2: {candidate2}")
    print(f"\np(c2|x) / p(c1|x) = {odds_base:.5f}")
    print(f"p(c2|x') / p(c1|x') = {odds_alt:.5f}")
    print(f"\nTOTAL Effect: (p(c2|x') / p(c1|x')) / (p(c2|x) / p(c1|x)) = {odds_ratio_alt:.3f}")

    for effect in effects:
        candidate1_probs, candidate2_probs = model.attention_intervention_experiment(intervention, effect)
        odds_intervention = candidate2_probs / candidate1_probs
        odds_ratio_intervention = odds_intervention / odds_base
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
        ax = sns.heatmap(odds_ratio_intervention, annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'{effect.capitalize()} Effect')


def report_interventions_summary(interventions, model, effects=('indirect', 'direct'), verbose=False):

    odds_ratio_sum = {}
    for effect in effects:
        odds_ratio_sum[effect] = torch.zeros((model.num_layers, model.num_heads))
    total_effect_sum = 0
    for intervention in interventions:
        x = intervention.base_strings_tok[0]
        x_alt = intervention.base_strings_tok[1]
        with torch.no_grad():
            candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples(x,
                                                                                              intervention.candidates_tok)
            candidate1_alt_prob, candidate2_alt_prob = model.get_probabilities_for_examples(x_alt,
                                                                                            intervention.candidates_tok)
            odds_base = candidate2_base_prob / candidate1_base_prob
            odds_alt = candidate2_alt_prob / candidate1_alt_prob
            odds_ratio = odds_alt / odds_base
            total_effect_sum += odds_ratio
            for effect in effects:
                candidate1_probs, candidate2_probs = model.attention_intervention_experiment(intervention, effect)
                odds_intervention = candidate2_probs / candidate1_probs
                odds_ratio_intervention = odds_intervention / odds_base
                odds_ratio_sum[effect] += odds_ratio_intervention

    n_interventions = len(interventions)
    print('*** SUMMARY ***')
    print(f"Num interventions: {n_interventions}")
    print(f"Mean total effect: {total_effect_sum / n_interventions:.2f}")

    for effect in effects:
        if verbose:
            print(f'\n{effect.upper()} Effect')
            if effect == 'indirect':
                print("   Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
                print(f"   Effect = (p(c2|x, Attn(x')) / p(c1|x, Attn(x')) / (p(c2|x) / p(c1|x))")
            elif effect == 'direct':
                print("   Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
                print(f"   Effect = (p(c2|x', Attn(x)) / p(c1|x', Attn(x)) / (p(c2|x) / p(c1|x))")
        plt.figure(figsize=(9, 7))
        ax = sns.heatmap(odds_ratio_sum[effect] / n_interventions, annot=True, annot_kws={"size": 12}, fmt=".2f")
        ax.set(xlabel='Head', ylabel='Layer', title=f'Mean {effect.capitalize()} Effect')

