import json

def main():

    models = ['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
    model_to_name = {
        'distilgpt2': 'GPT2-distil',
        'gpt2': 'GPT2-small',
        'gpt2-medium': 'GPT2-medium',
        'gpt2-large': 'GPT2-large',
        'gpt2-xl': 'GPT2-XL'
    }
    filters = ['filtered', 'unfiltered']
    stats = ['bergsma', 'bls']
    splits = ['dev', 'test']

    counts = {}

    # Process Winogender
    dataset = 'winogender'
    for filter in filters:
        for model_version in models:
            for stat in stats:
                fname =  f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
                    counts[(dataset, filter, model_version, stat)] = data['num_examples_analyzed']

    # Process Winogender
    dataset = 'winobias'
    for filter in filters:
        for model_version in models:
            for split in splits:
                fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                with open(fname) as f:
                    data = json.load(f)
                    counts[(dataset, filter, model_version, split)] = data['num_examples_analyzed']

    for model in models:
        print(f"{model_to_name[model]} & "
            f"{counts[('winobias', 'filtered', model, 'dev')]} & "
            f"{counts[('winobias', 'unfiltered', model, 'dev')]} & "
            f"{counts[('winobias', 'filtered', model, 'test')]} & "
            f"{counts[('winobias', 'unfiltered', model, 'test')]} & "
            f"{counts[('winogender', 'filtered', model, 'bls')]} & "
            f"{counts[('winogender', 'unfiltered', model, 'bls')]} & "
            f"{counts[('winogender', 'filtered', model, 'bergsma')]} & "
            f"{counts[('winogender', 'unfiltered', model, 'bergsma')]} \\\\")


if __name__ == '__main__':
    main()