import json
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer
from operator import itemgetter
from winobias import OCCUPATION_FEMALE_PCT
from collections import defaultdict
import math

def save_fig(prompts, heads, model, tokenizer, fname, device):
    plt.subplots_adjust(right=0.5)

    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(4, 4))
    axs[0].invert_yaxis()
    axs[0].tick_params(axis='y', which='major', pad=42)
    axs[0].yaxis.set_label_position("right")
    axs[0].yaxis.tick_right()
    plt.setp(axs[0].get_yticklabels(), fontsize=14, ha='center')
    axs[0].yaxis.set_ticks_position('none')

    attentions = []
    max_attn = 0
    for g_index in range(2):
        prompt = prompts[g_index]
        input_ = tokenizer.encode(prompt)
        batch = torch.tensor(input_).unsqueeze(0).to(device)
        attention = model(batch)[-1]
        seq_len = len(input_)
        attention = torch.stack(attention)
        attention = attention.squeeze(1)
        assert torch.allclose(attention.sum(-1), torch.tensor([1.0]))
        attentions.append(attention)
        if g_index == 0:
            seq = tokenizer.convert_ids_to_tokens(input_[:-1]) + ["she/he"]
            seq = [t.replace('Ġ', '') for t in seq]
        attn_sum = torch.Tensor([0])
        for layer, head in heads:
            attn_sum = attention[layer][head][-1] + attn_sum
        if max(attn_sum) > max_attn:
            max_attn = max(attn_sum)
    assert max_attn < 1
    xlim_upper = math.ceil(max_attn * 10) / 10
    for g_index in range(2):
        attention = attentions[g_index]
        left = 0
        prev = None
        head_names = []
        ax = axs[g_index]
        ax.set_xlim([0, xlim_upper])
        ax.set_xticks([0, xlim_upper])
        plts = []
        for i, (layer, head) in enumerate(heads):
            attn_last_word = attention[layer][head][-1].numpy()
            left += attn_last_word
            if i == 0:
                color = ['#4C71B0'] * seq_len
                color[0] = '#D1DBEB'
            else:
                color = ['#DD8452'] * seq_len
                color[0] = '#F8E3D8'
            if prev is None:
                p = ax.barh(seq, attn_last_word, color=color)
            else:
                p = ax.barh(seq, attn_last_word, left=prev, color=color)
            plts.append(p)
            head_names.append(f"Head {layer}-{head}")
            prev = attn_last_word
        if g_index == 0:
            ax.invert_xaxis()
        plt.setp(ax.get_xticklabels(), fontsize=12)
        sns.despine(left=True, bottom=True)
        if g_index == 1:
            leg = ax.legend(plts, head_names, fontsize=12, handlelength=.9, handletextpad=.4,
                            bbox_to_anchor=[0.1, 0.17])
            leg.legendHandles[0].set_color('#4C71B0')
            leg.legendHandles[1].set_color('#DD8452')

    axs[0].get_yticklabels()[0].set_color('#AFAFAF')

    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()


def main():
    sns.set_context("paper")
    sns.set_style("white")

    device = 'cpu'
    prompt =  "The nurse examined the farmer for injuries because"

    model_version = 'gpt2'
    heads = [(5, 10), (5, 8)]

    # Verify in dataset
    filter = 'filtered'
    split = 'dev'
    fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"

    with open(fname) as f:
        data = json.load(f)
    prompts = None
    results = data['results']
    # results_by_te = sorted(results, key=itemgetter('total_effect'), reverse=True)

    def get_female_ratio(prompt):
        female_pcts = []
        for occupation, female_pct in OCCUPATION_FEMALE_PCT.items():
            if occupation in prompt.lower():
                female_pcts.append(female_pct)
        assert len(female_pcts) == 2
        return max(female_pcts) / min(female_pcts)

    # results_by_ratio = sorted(results, key=lambda result: get_female_ratio(result['base_string1']), reverse=True)
    results_by_ratio = sorted(results, key=itemgetter('total_effect'), reverse=True)

    with torch.no_grad():
        # Get attention and validate
        model = GPT2Model.from_pretrained(model_version, output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model.eval()
        for result_index, result in enumerate(results_by_ratio):
            prompts = (result['base_string1'], result['base_string2'])
            fname = f'results/attention_intervention/qualitative/winobias_{model_version}_{filter}_{split}_{result_index}.pdf'
            save_fig(prompts, heads, model, tokenizer, fname, device)


if __name__ == '__main__':
    main()