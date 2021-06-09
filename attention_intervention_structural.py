"""Performs attention intervention on Winobias samples and saves results to JSON file."""

import json
import os
import random

import sys
from pandas import DataFrame
from transformers import (
    GPT2Tokenizer, TransfoXLTokenizer, XLNetTokenizer
)

from attention_utils import perform_interventions, get_odds_ratio
from experiment_num_agreement import Model, Intervention

from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
import vocab_utils as vocab

def construct_templates(attractor):
    templates = []
    if attractor in  ['singular', 'plural']:
        for p in get_prepositions():
            for ppns, ppnp in get_preposition_nouns():
                ppn = ppns if attractor == 'singular' else ppnp
                template = ' '.join(['The', '{}', p, 'the', ppn])
                templates.append(template)
    elif attractor in ('rc_singular', 'rc_plural', 'rc_singular_no_that', 'rc_plural_no_that'):
        for noun2s, noun2p in get_nouns2():
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in get_verbs2():
                verb2 = verb2s if attractor.startswith('rc_singular') else verb2p
                if attractor.endswith('no_that'):
                    template = ' '.join(['The', '{}', 'the', noun2, verb2])
                else:
                    template = ' '.join(['The', '{}', 'that', 'the', noun2, verb2])
                    # templates.append(' '.join(['The', '{}', 'that', 'the', noun2s, verb2s]))
                    # templates.append(' '.join(['The', '{}', 'that', 'the', noun2p, verb2p]))
                templates.append(template)
    elif attractor in ('within_rc_singular', 'within_rc_plural', 'within_rc_singular_no_that', 'within_rc_plural_no_that'):
        for ns, np in vocab.get_nouns():
            noun = ns if attractor.startswith('within_rc_singular') else np
            if attractor.endswith('no_that'):
                template = ' '.join(['The', noun, 'the', '{}'])
            else:
                template = ' '.join(['The', noun, 'that', 'the', '{}'])
                # templates.append(' '.join(['The', ns, 'that', 'the', '{}']))
                # templates.append(' '.join(['The', np, 'that', 'the', '{}']))
            templates.append(template)
    elif attractor == 'distractor':
        for  adv1 in  get_adv1s():
            for adv2 in get_adv2s():
                templates.append(' '.join(['The', '{}', adv1, 'and', adv2]))
    elif attractor == 'distractor_1':
        for adv1 in get_adv1s():
            templates.append(' '.join(['The', '{}', adv1]))

    else:
        templates = ['The {}']
    return templates

def load_structural_interventions(tokenizer, device, attractor, seed, examples):
    # build list of interventions
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates(attractor)
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in get_nouns2():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, noun2s, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            [noun2s, noun2p],
                            [v_singular, v_plural],
                            device=device)
                        used_word_count += 1
                    except Exception as e:
                        pass
        else:
            for ns, np in vocab.get_nouns():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try: 
                        intervention_name = '_'.join([temp, ns, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            [ns, np],
                            [v_singular, v_plural],
                            device=device)
                        used_word_count += 1
                    except Exception as e:
                        pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:
        random.seed(seed)
        interventions = {k: v
                for k, v in random.sample(interventions.items(), examples)}
    return interventions


def get_interventions_structural(gpt2_version, do_filter, model, tokenizer,
                                 device='cuda', filter_quantile=0.25, seed=3, attractor=None, examples=100):
    interventions = load_structural_interventions(tokenizer, device, attractor, seed, examples)
    intervention_list = [intervention for intervention in interventions.values()]
    interventions = intervention_list
    
    json_data = {'model_version': gpt2_version,
            'do_filter': do_filter,
            'num_examples_loaded': len(interventions)}
    if do_filter:
        df = DataFrame({'odds_ratio': [get_odds_ratio(intervention, model) for intervention in intervention_list]})
        df_expected = df[df.odds_ratio > 1]
        threshold = df_expected.odds_ratio.quantile(filter_quantile)
        filtered_interventions = []
        assert len(intervention_list) == len(df)
        for i in range(len(intervention_list)):
            intervention = intervention_list[i]
            odds_ratio = df.iloc[i].odds_ratio
            if odds_ratio > threshold:
                filtered_interventions.append(intervention)

        print(f'Num examples with odds ratio > 1: {len(df_expected)} / {len(intervention_list)}')
        print(
            f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_interventions)} / {len(intervention_list)}')
        json_data['num_examples_aligned'] = len(df_expected)
        json_data['filter_quantile'] = filter_quantile
        json_data['threshold'] = threshold
        interventions = filtered_interventions
    json_data['num_examples_analyzed'] = len(interventions)
    return interventions, json_data


def intervene_attention(gpt2_version, do_filter, attractor, device='cuda', filter_quantile=0.25, examples=100,\
        seed=3, random_weights=False):

    model = Model(output_attentions=True, gpt2_version=gpt2_version,
                   device=device, random_weights=random_weights)
    tokenizer = (GPT2Tokenizer if model.is_gpt2 else
                  TransfoXLTokenizer if model.is_txl else
                  # XLNetTokenizer if model.is_xlnet
                  XLNetTokenizer
                  ).from_pretrained(gpt2_version)

    interventions, json_data = get_interventions_structural(gpt2_version, do_filter,
                                                            model, tokenizer,
                                                            device, filter_quantile,
                                                            seed=seed, attractor=attractor,
                                                            examples=examples)

    results = perform_interventions(interventions, model)
    json_data['mean_total_effect'] = DataFrame(results).total_effect.mean()
    json_data['mean_model_indirect_effect'] = DataFrame(results).indirect_effect_model.mean()
    json_data['mean_model_direct_effect'] = DataFrame(results).direct_effect_model.mean()
    filter_name = 'filtered' if do_filter else 'unfiltered'
    if random_weights:
        gpt2_version += '_random'
    fname = f"attention_results/{attractor}/attention_intervention_{gpt2_version}_{filter_name}.json"
    base_path = '/'.join(fname.split('/')[:-1])
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    json_data['results'] = results
    with open(fname, 'w') as f:
        json.dump(json_data, f)

if __name__ == "__main__":
    model = sys.argv[1]
    device = sys.argv[2]
    filter_quantile = float(sys.argv[3])
    random_weights = sys.argv[4] == 'random'
    attractor = sys.argv[5]
    seed = int(sys.argv[6])
    examples = int(sys.argv[7])
    #intervene_attention(model, True, attractor, device=device, filter_quantile=filter_quantile, examples=examples, \
    #        seed=seed, random_weights=random_weights)
    intervene_attention(model, False, attractor, device=device, filter_quantile=0.0, examples=examples, seed=seed, \
            random_weights=random_weights)
