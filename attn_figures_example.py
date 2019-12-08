import json
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from transformers import GPT2Model, GPT2Tokenizer
from operator import itemgetter
import math
import numpy as np
import matplotlib.patches as patches

BLACK = '#000000'
GRAY = '#303030'

def save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices=None):
    palette = sns.color_palette('muted')
    # plt.subplots_adjust(bottom=.2)
    plt.rc('text', usetex=True)
    # plt.rcParams["axes.edgecolor"] = "black"
    # plt.rcParams["axes.linewidth"] = 1
    fig, axs = plt.subplots(1, 2, sharey=False, figsize=(3.3, 1.95))
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
        left = None
        for i, (layer, head) in enumerate(heads):
            attn_last_word = attention[layer][head][-1].numpy()

            if left is None:
                p = ax.barh(formatted_seq, attn_last_word, color=palette[i], linewidth=0)
            else:
                p = ax.barh(formatted_seq, attn_last_word, left=left, color=palette[i], linewidth=0)
            if left is None:
                left = np.zeros_like(attn_last_word)
            left += attn_last_word

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

        ax.set_xlim([0, xlim_upper])
        ax.set_xticks([0, xlim_upper])
        ax.invert_yaxis()
        plt.setp(ax.get_yticklabels(), fontsize=8, ha='right')
        ax.set_xticks([0, 0.5])
        plt.setp(ax.get_xticklabels(), fontsize=7)
        sns.despine(left=True, bottom=True)
        ax.tick_params(axis='x', pad=0, length=0)
        ax.tick_params(axis='y', pad=0)
        # if g_index == 0:
        #     ax.legend(plts, head_names, ncol=3, fontsize=12, handlelength=.9, handletextpad=.3, labelspacing = 0.15,
        #               borderpad=0.2, loc='upper center')#, bbox_to_anchor=[.4, .025], bbox_transform=plt.gcf().transFigure)
        #               #)bbox_to_anchor=[0.1, 0.17])

        ax.yaxis.labelpad = 0
        ax.xaxis.labelpad = 0


    # fig.legend(handles, labels, loc='upper center')
    # plt.figlegend(plts, head_names,(1.04, 0), ncol=3, fontsize=12, handlelength=.9, handletextpad=.3)#, bbox_to_anchor=(0.5, 1.05))
    lgd = plt.figlegend(plts, head_names,'lower center', fontsize=7, borderpad=0.5, handlelength=.9,
                        handletextpad=.3, labelspacing = 0.15, bbox_to_anchor=(0.86, 0.11))
    # plt.tight_layout()
    # rect = patches.Rectangle((4, 4), 4, 4, linewidth=2, edgecolor='r', facecolor='none')
    #
    # # Add the patch to the Axes
    # ax = plt.gca()
    # ax.add_patch(rect)

    plt.savefig(fname, format='pdf', bbox_extra_artists = (lgd,), bbox_inches = 'tight')


    plt.close()


def main():
    sns.set_context("paper")
    sns.set_style("white")
    device = 'cpu'

    top_heads = {
        'gpt2':[(5, 8), (5, 10), (4,6)],
        'gpt2-medium': [(10, 9), (6, 15), (10,12)],
        'gpt2-xl':[(16,15), (16, 24), (17,10)],
        'gpt2-large':[(16,19), (16,5), (15,6)],
        'distilgpt2': [(3,1), (2,6), (3,6)]
    }
    models = ['gpt2', 'gpt2-medium', 'gpt2-xl', 'gpt2-large', 'distilgpt2']

    split = 'dev'
    testing = False
    for model_version in models:
        heads = top_heads[model_version]
        if model_version == 'distilgpt2':
            filter = 'unfiltered' # In order to get canonical example
        else:
            filter = 'filtered'
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

                if "The nurse examined the farmer for injuries because"in prompts[0]:
                    highlight_indices = [1, 4]
                    fname = f'results/attention_intervention/qualitative/winobias_{model_version}_main.pdf'
                    save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices)
                else:
                    highlight_indices = None
                fname = f'results/attention_intervention/qualitative/winobias_{model_version}_{filter}_{split}_{result_index}.pdf'
                save_fig(prompts, heads, model, tokenizer, fname, device, highlight_indices)
                # For testing only:
                if testing:
                    break
        if testing:
            break


if __name__ == '__main__':
    main()