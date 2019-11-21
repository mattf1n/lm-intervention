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

    effects = {}

    # Process Winogender
    dataset = 'winogender'
    for filter in filters:
        for model_version in models:
            for stat in stats:
                fname =  f"winogender_data/attention_intervention_{stat}_{model_version}_{filter}.json"
                with open(fname) as f:
                    data = json.load(f)
                    effects[(dataset, filter, model_version, stat)] = data['mean_total_effect']

    # Process Winogender
    dataset = 'winobias'
    for filter in filters:
        for model_version in models:
            for split in splits:
                fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
                with open(fname) as f:
                    data = json.load(f)
                    effects[(dataset, filter, model_version, split)] = data['mean_total_effect']

    for model in models:
        print(f"{model_to_name[model]} & "
            f"{effects[('winobias', 'filtered', model, 'dev')]:.3f} & "
            f"{effects[('winobias', 'unfiltered', model, 'dev')]:.3f} & "
            f"{effects[('winobias', 'filtered', model, 'test')]:.3f} & "
            f"{effects[('winobias', 'unfiltered', model, 'test')]:.3f} & "
            f"{effects[('winogender', 'filtered', model, 'bls')]:.3f} & "
            f"{effects[('winogender', 'unfiltered', model, 'bls')]:.3f} & "
            f"{effects[('winogender', 'filtered', model, 'bergsma')]:.3f} & "
            f"{effects[('winogender', 'unfiltered', model, 'bergsma')]:.3f} \\\\")


if __name__ == '__main__':
    main()