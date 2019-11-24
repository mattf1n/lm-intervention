import json
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer
from operator import itemgetter
import math

BLACK = '#000000'
GRAY = '#303030'

def save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices=None):
    palette = sns.color_palette('muted')
    plt.subplots_adjust(right=0.5)
    plt.rc('text', usetex=True)
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(5, 4))
    axs[0].yaxis.set_ticks_position('none')
    plt.rcParams.update({'axes.titlesize': 'xx-large'})
    attentions = []
    max_attn = 0
    seqs = []
    for g_index in range(2):
        prompt = prompts[g_index]
        input_ = tokenizer.encode(prompt)
        batch = torch.tensor(input_).unsqueeze(0).to(device)
        attention = model(batch)[-1]
        seq = tokenizer.convert_ids_to_tokens(input_)
        seq = [t.replace('Ġ', '') for t in seq]
        seqs.append(seq)
        seq_len = len(input_)
        attention = torch.stack(attention)
        attention = attention.squeeze(1)
        assert torch.allclose(attention.sum(-1), torch.tensor([1.0]))
        attentions.append(attention)
        attn_sum = torch.Tensor([0])
        for layer, head in heads:
            attn_sum = attention[layer][head][-1] + attn_sum
        if max(attn_sum) > max_attn:
            max_attn = max(attn_sum)
    xlim_upper = math.ceil(max_attn * 10) / 10
    for g_index in range(2):
        attention = attentions[g_index]
        left = 0
        prev = None
        head_names = []
        ax = axs[g_index]
        seq = seqs[g_index]
        formatted_seq = []
        if highlight_indices:
            for i, t in enumerate(seq):
                if i in highlight_indices:
                    if i == highlight_indices[g_index]:
                        t = f"\\textbf{{{t}}}"
                    else:
                        t = f"\\underline{{{t}}}"
                formatted_seq.append(t)
            formatted_seq[-1] = f"\\textbf{{{formatted_seq[-1]}}}"
        else:
            formatted_seq = seq

        plts = []
        for i, (layer, head) in enumerate(heads):
            attn_last_word = attention[layer][head][-1].numpy()
            left += attn_last_word
            if prev is None:
                p = ax.barh(formatted_seq, attn_last_word, color=palette[i])
            else:
                p = ax.barh(formatted_seq, attn_last_word, left=prev, color=palette[i])
            if highlight_indices:
                for i in range(seq_len):
                    if highlight_indices[g_index] == i:
                        color = BLACK
                    else:
                        color = GRAY
                    ax.get_yticklabels()[i].set_color(color)
                ax.get_yticklabels()[-1].set_color(BLACK)
            plts.append(p)
            head_names.append(f"Head {layer}-{head}")
            prev = attn_last_word

        ax.set_xlim([0, xlim_upper])
        ax.set_xticks([0, xlim_upper])
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=14, ha='right')
        ax.set_xticks([0, 0.5])
        plt.setp(ax.get_xticklabels(), fontsize=12)
        sns.despine(left=True, bottom=True)
        if g_index == 1:
            ax.legend(plts, head_names, fontsize=12, handlelength=.9, handletextpad=.4, bbox_to_anchor=[0.1, 0.17])

    plt.tight_layout()
    plt.savefig(fname, format='pdf')
    plt.close()


def main():
    sns.set_context("paper")
    sns.set_style("white")
    device = 'cpu'

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
    results_by_ratio = sorted(results, key=itemgetter('total_effect'), reverse=True)

    with torch.no_grad():
        # Get attention and validate
        model = GPT2Model.from_pretrained(model_version, output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model.eval()
        for result_index, result in enumerate(results_by_ratio):
            prompts = (result['base_string1'], result['base_string2'])
            fname = f'results/attention_intervention/qualitative/winobias_{model_version}_{filter}_{split}_{result_index}.pdf'
            if "The nurse examined the farmer for injuries because"in prompts[0]:
                highlight_indices = [1, 4]
            else:
                highlight_indices = None
            save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices)


if __name__ == '__main__':
    main()