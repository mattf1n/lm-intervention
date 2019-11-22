import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from attention_utils import topk_indices
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
import os
from experiment import Model
from transformers import GPT2Model, GPT2Tokenizer
import torch

def main():
    sns.set_context("paper")
    sns.set_style("white")

    device = 'cpu'
    prompt =  "The nurse examined the farmer for injuries because"


    model_version = 'gpt2'; num_layers=12; num_heads=12

    # Verify in dataset
    filter = 'filtered'
    split = 'dev'
    fname = f"winobias_data/attention_intervention_{model_version}_{filter}_{split}.json"
    with open(fname) as f:
        data = json.load(f)
    prompts = None
    for result in data['results']:
        if result['base_string1'] == prompt + ' she':
            prompts = (result['base_string1'], result['base_string2'])
            break
    assert prompts is not None

    with torch.no_grad():
        # Get attention and validate
        model = GPT2Model.from_pretrained(model_version, output_attentions=True)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model.eval()
        attentions = []
        for prompt in prompts:
            input_ = tokenizer.encode(prompt)
            batch = torch.tensor(input_).unsqueeze(0).to(device)
            attention = model(batch)[-1]
            batch_size = 1
            seq_len = len(input_)
            attention = torch.stack(attention)
            assert attention.shape == (num_layers, batch_size, num_heads, seq_len, seq_len)
            attention = attention.squeeze(1)
            assert attention.shape == (num_layers, num_heads, seq_len, seq_len)
            assert torch.allclose(attention.sum(-1), torch.tensor([1.0]))
            attentions.append(attention)

        seq = tokenizer.convert_ids_to_tokens(input_[:-1]) + ["she/he"]
        seq = [t.replace('Ġ', '') for t in seq]
        print(seq)
        width = 0.6
        sns.set_color_codes("pastel")
        heads = [(5,10), (5, 8)]
        f, ax = plt.subplots(figsize=(5, 5))
        left = 0
        prev = None
        plts = []
        head_names = []
        for i, (layer, head) in enumerate(heads):
            attn_last_word = attention[layer][head][-1].numpy()
            # if prev is not None:
            #     ax = plt.barh(seq, attn_last_word, width, bottom=prev)
            # else:
            #     plt.barh(seq, attn_last_word, width)
            left += attn_last_word
            if prev is None:
                p = plt.barh(seq, attn_last_word)
            else:
                p = plt.barh(seq, attn_last_word, left=prev)
            plts.append(p)
            head_names.append(f"Head {layer}-{head}")
            prev = attn_last_word

        ax.invert_yaxis()
        # ax.set_yticklabels(seq, ha='center', minor=False, size=14)
        ax.tick_params(axis='y', which='major', pad=30)
        ax.legend(plts, head_names, loc='lower right', fontsize=14)
        plt.setp(ax.get_yticklabels(), fontsize=14, ha='center')
        plt.setp(ax.get_xticklabels(), fontsize=14)
        sns.despine(left=True, bottom=True)
        plt.show()





if __name__ == '__main__':
    main()