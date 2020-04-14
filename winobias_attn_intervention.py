from attention_utils import perform_interventions, get_odds_ratio
import fire
import winobias
from experiment import Model
from transformers import GPT2Tokenizer
from transformers import TransfoXLTokenizer ### NEW
from datetime import datetime ### NEW
import os ### NEW
import json
from pandas import DataFrame


def get_interventions_winobias(gpt2_version, do_filter, split, model, tokenizer,
                                device='cuda', filter_quantile=0.25):
    if split == 'dev':
        examples = winobias.load_dev_examples()
    elif split == 'test':
        examples = winobias.load_test_examples()
    else:
        raise ValueError(f"Invalid split: {split}")
    json_data = {'model_version': gpt2_version,
            'do_filter': do_filter,
            'split': split,
            'num_examples_loaded': len(examples)}
    if do_filter:
        interventions = [ex.to_intervention(tokenizer) for ex in examples]
        df = DataFrame({'odds_ratio': [get_odds_ratio(intervention, model) for intervention in interventions]})
        df_expected = df[df.odds_ratio > 1]
        threshold = df_expected.odds_ratio.quantile(filter_quantile)
        filtered_examples = []
        assert len(examples) == len(df)
        for i in range(len(examples)):
            ex = examples[i]
            odds_ratio = df.iloc[i].odds_ratio
            if odds_ratio > threshold:
                filtered_examples.append(ex)

        print(f'Num examples with odds ratio > 1: {len(df_expected)} / {len(examples)}')
        print(
            f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_examples)} / {len(examples)}')
        json_data['num_examples_aligned'] = len(df_expected)
        json_data['filter_quantile'] = filter_quantile
        json_data['threshold'] = threshold
        examples = filtered_examples
    json_data['num_examples_analyzed'] = len(examples)
    interventions = [ex.to_intervention(tokenizer) for ex in examples]
    return interventions, json_data

def intervene_attention(gpt2_version, do_filter, split, device='cuda', filter_quantile=0.25, random_weights=False):
    model = Model(output_attentions=True, gpt2_version=gpt2_version, device=device, random_weights=random_weights)
    ### New ###
    if model.is_txl:
        print('****** NEW: Using TransoXL tokenizer')
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)
    ### New ###

    interventions, json_data = get_interventions_winobias(gpt2_version, do_filter, split, model, tokenizer,
                                                            device, filter_quantile)
    results = perform_interventions(interventions, model)
    json_data['mean_total_effect'] = DataFrame(results).total_effect.mean()
    json_data['mean_model_indirect_effect'] = DataFrame(results).indirect_effect_model.mean()
    json_data['mean_model_direct_effect'] = DataFrame(results).direct_effect_model.mean()
    filter_name = 'filtered' if do_filter else 'unfiltered'
    if random_weights:
        gpt2_version += '_random'
    ### New ###
    if model.is_txl:
        # fname = f"txl_results/attention_intervention/attention_intervention_{gpt2_version}_{filter_name}_{split}.json"
        dt_string = datetime.now().strftime('%Y%m%d')
        folder_name = dt_string + '_attention_intervention'
        base_path = os.path.join('txl_results/attention_intervention', folder_name)
        if not os.path.exists(base_path): os.makedirs(base_path)
        fname = os.path.join(base_path, f'winobias_{gpt2_version}_{filter_name}_{split}.json')
    else:
        fname = f"winobias_data/attention_intervention_{gpt2_version}_{filter_name}_{split}.json"
    ### New ###
    json_data['results'] = results
    with open(fname, 'w') as f:
        json.dump(json_data, f)


if __name__ == "__main__":
    fire.Fire(intervene_attention)
