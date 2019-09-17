import torch
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

def report_intervention(intervention, effect, model, log_scale=False):
    candidate1_base_prob, candidate2_base_prob, candidate1_alt_prob, candidate2_alt_prob, candidate1_probs, candidate2_probs = model.attention_intervention_experiment(
        intervention, effect)

    candidate1 = ' '.join(intervention.candidates[0]).replace('Ġ', '')
    candidate2 = ' '.join(intervention.candidates[1]).replace('Ġ', '')

    odds_base = candidate2_base_prob / candidate1_base_prob
    odds_intervention = candidate2_probs / candidate1_probs
    odds_ratio_intervention = odds_intervention / odds_base
    odds_alt = candidate2_alt_prob / candidate1_alt_prob
    odds_ratio_alt = odds_alt / odds_base

    if log_scale:
        odds_base = torch.log(odds_base)
        odds_intervention = torch.log(odds_intervention)
        odds_ratio_intervention = torch.log(odds_ratio_intervention)
        odds_alt = torch.log(odds_alt)
        odds_ratio_alt = torch.log(odds_ratio_alt)

    print('Effect:', effect.upper())
    if effect == 'indirect':
        print("Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
    elif effect == 'direct':
        print("Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
    else:
        raise ValueError(f"Invalid effect: {effect}")
    print(f"x : {intervention.base_strings[0]}")
    print(f"x': {intervention.base_strings[1]}")
    print("Input value: x")
    print(f"Continuations compared: {candidate1} / {candidate2}")
    if log_scale:
        print("*** All values are on LOG scale ***")
    print(f"p({candidate2}|x) / p({candidate1}|x) = {odds_base:.5f}")
    print(f"p({candidate2}|x') / p({candidate1}|x') = {odds_alt:.5f}")
    print(
        f"Total Effect: p({candidate2}|x') / p({candidate1}|x') / p({candidate2}|x) / p({candidate1}|x) = {odds_ratio_alt:.3f}")
    if effect == 'indirect':
        print(
            f'Indirect Effect (p("{candidate2}"|x, Attn(x\')) / p("{candidate1}"|x, Attn(x\')) / (p("{candidate2}"|x) / p("{candidate1}"|x)):')
    elif effect == 'direct':
        print(
            f'Direct Effect: (p("{candidate2}"|x\', Attn(x)) / p("{candidate1}"|x\', Attn(x) / (p("{candidate2}"|x) / p("{candidate1}"|x)) =')

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(odds_ratio_intervention, annot=True, annot_kws={"size": 12}, fmt=".2f")
    ax.set(xlabel='Head', ylabel='Layer')


def report_interventions(interventions, effect, model, log_scale=False):
    odds_ratio_intervention_sum = torch.zeros((model.num_layers, model.num_heads))
    odds_ratio_alt_sum = 0

    for intervention in interventions:
        candidate1_base_prob, candidate2_base_prob, candidate1_alt_prob, candidate2_alt_prob, candidate1_probs, candidate2_probs = model.attention_intervention_experiment(
            intervention, effect)
        odds_base = candidate2_base_prob / candidate1_base_prob
        odds_intervention = candidate2_probs / candidate1_probs
        odds_ratio_intervention = odds_intervention / odds_base
        if log_scale:
            odds_ratio_intervention = torch.log(odds_ratio_intervention)
        odds_ratio_intervention_sum += odds_ratio_intervention

        odds_alt = candidate2_alt_prob / candidate1_alt_prob
        odds_ratio_alt = odds_alt / odds_base
        if log_scale:
            odds_ratio_alt = torch.log(odds_ratio_alt)
        odds_ratio_alt_sum += odds_ratio_alt

    mean_odds_ratio_intervention = odds_ratio_intervention_sum / len(interventions)
    mean_odds_ratio_alt = odds_ratio_alt_sum / len(interventions)

    print('*** SUMMARY ***')
    print('Effect:', effect.upper())
    if effect == 'indirect':
        print("Intervention: replace Attn(x) with Attn(x') in a specific layer/head")
    elif effect == 'direct':
        print("Intervention: replace x with x' while preserving Attn(x) in a specific layer/head")
    else:
        raise ValueError(f"Invalid effect: {effect}")
    if log_scale:
        print("*** All values are LOG values ***")
    print(f"Num interventions: {len(interventions)}")
    print(f"Mean total effect: {mean_odds_ratio_alt:.2f}")
    print(f"Mean {effect} effect:")
    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(mean_odds_ratio_intervention, annot=True, annot_kws={"size": 12}, fmt=".2f")
    ax.set(xlabel='Head', ylabel='Layer')