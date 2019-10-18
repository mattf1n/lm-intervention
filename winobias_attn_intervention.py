import fire
import winobias
from experiment import Model
from attention_utils import perform_interventions
from transformers import GPT2Tokenizer
import json


def intervene_attention(gpt2_version, do_filter, device='cuda', filter_quantile=0.25):
    examples = winobias.load_dev_examples()
    model = Model(output_attentions=True, gpt2_version=gpt2_version, device=device)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)
    if do_filter:
        df = winobias.analyze(examples, gpt2_version=gpt2_version)
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
        print(f'Num examples with odds ratio > {threshold:.4f} ({filter_quantile} quantile): {len(filtered_examples)} / {len(examples)}')

        examples = filtered_examples
    interventions = [ex.to_intervention(tokenizer) for ex in examples]
    results = perform_interventions(interventions, model)
    filter_name = 'filtered' if do_filter else 'unfiltered'
    fname = f"winobias_data/attention_intervention_results_{gpt2_version}_{filter_name}.json"
    with open(fname, 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(intervene_attention)

