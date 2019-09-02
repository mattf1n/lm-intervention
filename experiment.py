
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
from functools import partial
from tqdm import tqdm
# from tqdm import tqdm_notebook
import math

from collections import Counter, defaultdict

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer

# sns.set(style="ticks", color_codes=True)

np.random.seed(1)
torch.manual_seed(1)


class Intervention():
    '''
    Wrapper for all the possible interventions
    '''
    def __init__(self,
                 tokenizer,
                 base_string: str,
                 substitutes: list,
                 candidates: list,
                 device='cpu'):
        super()
        self.device = device
        self.enc = tokenizer
        # All the initial strings
        # First item should be neutral, others tainted
        self.base_strings = [base_string.format(s)
                             for s in substitutes]
        # Tokenized bases
        self.base_strings_tok = [self._to_batch(s)
                                 for s in self.base_strings]
        # Where to intervene
        self.position = base_string.split().index('{}')

        # How to extend the string
        self.candidates = ['Ġ' + c for c in candidates]
        # tokenized candidates
        self.candidates_tok = [self.enc.convert_tokens_to_ids(c)
                               for c in self.candidates]

    def _to_batch(self, txt):
        encoded = self.enc.encode(txt)
        return torch.tensor(encoded, dtype=torch.long)\
                    .unsqueeze(0)\
                    .repeat(1, 1).to(device=self.device)


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self, device='cpu', output_attentions=False):
        super()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=output_attentions)
        self.model.eval()
        self.model.to(device)

        # Options
        self.top_k = 5
        # 12 for GPT-2
        self.num_layers = len(self.model.transformer.h)
        # 768 for GPT-2
        self.num_neurons = self.model.transformer.wte.weight.shape[1]
        # 12 for GPT-2
        self.num_heads = self.model.transformer.h[0].attn.n_head

        # multiplier for intervention; needs to be pretty large (~100) to see a change
        # TODO: plot the intervention results (how many neurons are flipped) for different alphas
        self.alpha = 500

    def get_representations(self, context, position):
        # Hook for saving the representation
        def extract_representation_hook(module, input, output, position, representations, layer):
            representations[layer] = output[0][position]
        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            for layer in range(self.num_layers):
                handles.append(self.model.transformer.h[0]\
                                   .mlp.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=layer)))
            logits, past = self.model(context)
            for h in handles:
                h.remove()
        # print(representation[0][:5])
        return representation

    def get_probabilities_for_examples(self, context, outputs):
        logits, past = self.model(context)[:2]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[0][outputs]

    def neuron_intervention(self,
                            context,
                            outputs,
                            repr_difference,
                            layer,
                            neuron,
                            position):
        # Hook for changing representation during forward pass
        def intervention_hook(module, input, output, position, neuron, intervention):
            output[0][position][neuron] += intervention

        intervention_rep = self.alpha * repr_difference[layer][neuron]
        mlp_intervention_handle = self.model.transformer.h[layer]\
                                      .mlp.register_forward_hook(
            partial(intervention_hook,
                    position=position,
                    neuron=neuron,
                    intervention=intervention_rep))
        new_probabilities = self.get_probabilities_for_examples(
            context,
            outputs)
        mlp_intervention_handle.remove()
        return new_probabilities

    def head_pruning_intervention(self,
                                  context,
                                  outputs,
                                  layer,
                                  head):
        # Recreate model and prune head
        save_model = self.model
        # TODO Make this more efficient
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.prune_heads({layer: [head]})
        self.model.eval()

        # Compute probabilities without head
        new_probabilities = self.get_probabilities_for_examples(
            context,
            outputs)

        # Reinstate original model
        # TODO Handle this in cleaner way
        self.model = save_model

        return new_probabilities

    def attention_intervention(self,
                               context,
                               outputs,
                               layer,
                               attn_override,
                               attn_override_mask):
        """ Override attention values in specified layer

        Args:
            context: context text
            layer: layer in which to intervene
            attn_override: values to override the computed attention weights. Shape is [num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override. Shape is [num_heads, seq_len, seq_len]
        """

        def intervention_hook(module, input, outputs):
            outputs[0] = self.get_attention_output(input[0], module, attn_override, attn_override_mask)

        with torch.no_grad():
            hook = self.model.transformer.h[layer].attn.register_forward_hook(intervention_hook)
            new_probabilities = self.get_probabilities_for_examples(
                context,
                outputs)
            hook.remove()
            return new_probabilities

    def get_attention_output(self, x, attn_obj, attn_override, attn_override_mask):
        """Get the output from `Attention` module, but with overridden attention weights. This applies to a single
            transformer layer.

        Args:
            x: input text
            layer: layer to override attention
            attn_override: values to override the computed attention weights. Shape is [num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention values to override. Shape is [num_heads, seq_len, seq_len]
        """

        # Following code is based on `Attention.forward` from `modeling_gpt2.py`. However:
        #    - Does not support following arguments to `Attention.forward`: `layer_past`, `head_mask`
        #    - Does not support `output_attentions` configuration option
        #    - Assumes in eval mode (e.g. does not apply dropout)
        x = attn_obj.c_attn(x)
        query, key, value = x.split(attn_obj.split_size, dim=2)
        query = attn_obj.split_heads(query)
        key = attn_obj.split_heads(key, k=True)
        value = attn_obj.split_heads(value)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking

        # Following is based on Attention._attn
        w = torch.matmul(query, key)
        if attn_obj.scale:
            w = w / math.sqrt(value.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = attn_obj.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)
        w = nn.Softmax(dim=-1)(w)

        # Override attention weights where mask is 1, else keep original values
        w = torch.where(attn_override_mask, attn_override, w)

        # Apply attention weights to compute outputs
        a = torch.matmul(w, value)
        a = attn_obj.merge_heads(a)
        a = attn_obj.c_proj(a)

        return a

    def neuron_intervention_experiment(self, word2intervention):
        """
        run multiple intervention experiments
        """

        word2intervention_results = {}
        for word in tqdm(word2intervention, desc='words'):
            word2intervention_results[word] = self.neuron_intervention_single_experiment(word2intervention[word])

        return word2intervention_results

    def neuron_intervention_single_experiment(self, intervention):
        """
        run one full neuron intervention experiment
        """

        # Bookkeeping
        layer_to_candidate1, layer_to_candidate2 = Counter(), Counter()
        layer_to_candidate1_probs, layer_to_candidate2_probs = defaultdict(list), defaultdict(list)

        with torch.no_grad():
            '''
            Compute representations for base terms (one for each side of bias)
            '''
            man_representations = self.get_representations(
                intervention.base_strings_tok[1],
                intervention.position)
            woman_representations = self.get_representations(
                intervention.base_strings_tok[2],
                intervention.position)
            representation_difference = {k: v - woman_representations[k]
                                         for k, v in man_representations.items()}
            '''
            Now intervening on potentially biased example
            '''
            context = intervention.base_strings_tok[0]
            '''
            Probabilities without intervention (Base case)
            '''
            base_probs = self.get_probabilities_for_examples(
                context,
                intervention.candidates_tok)
            print("Base case: {} ____".format(intervention.base_strings[0]))
            for token, prob in zip(intervention.candidates, base_probs):
                print("{}: {:.2f}%".format(token, prob*100))

            '''
            Intervene at every possible neuron
            '''
            for layer in tqdm(range(self.num_layers)):
                for neuron in range(self.num_neurons):
                    candidate1_prob, candidate2_prob = self.neuron_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        repr_difference=representation_difference,
                        layer=layer,
                        neuron=neuron,
                        position=intervention.position)

                    layer_to_candidate1_probs[layer].append(candidate1_prob)
                    layer_to_candidate2_probs[layer].append(candidate2_prob)
                    if candidate1_prob > candidate2_prob:
                        layer_to_candidate1[layer] += 1
                    else:
                        layer_to_candidate2[layer] += 1
        return (layer_to_candidate1,
                layer_to_candidate1_probs,
                layer_to_candidate2,
                layer_to_candidate2_probs)

    def attention_intervention_experiment(self, intervention):
        """
        Run one full attention intervention experiment measuring indirect effect.
        """
        x = intervention.base_strings_tok[0] # E.g. The doctor asked the nurse a question. He
        x_alt = intervention.base_strings_tok[1] # E.g. The doctor asked the nurse a question. She

        attention_override = self.model(x_alt)[-1] # Get attention for x_alt
        batch_size = 1
        seq_len = x.shape[1]
        seq_len_alt = x_alt.shape[1]
        assert seq_len == seq_len_alt
        assert len(attention_override) == self.num_layers
        assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len)

        with torch.no_grad():
            candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
                x,
                intervention.candidates_tok)

            candidate1_probs = torch.zeros((self.num_layers, self.num_heads))
            candidate2_probs = torch.zeros((self.num_layers, self.num_heads))
            # Intervene at every head by overlaying attention induced by x_alt
            for layer in tqdm(range(self.num_layers)):
                layer_attention_override = attention_override[layer]
                for head in tqdm(range(self.num_heads)):
                    attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                    attention_override_mask[0][head] = 1 # Set mask for head only
                    candidate1_probs[layer][head], candidate2_probs[layer][head] = self.attention_intervention(
                        context=x,
                        outputs=intervention.candidates_tok,
                        layer=layer,
                        attn_override=layer_attention_override,
                        attn_override_mask=attention_override_mask)

        return candidate1_base_prob, candidate2_base_prob, candidate1_probs, candidate2_probs

def main():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model()

    base_sentence = "The {} said that"
    biased_word = "teacher"
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"])

    c1, c1_probs, c2, c2_probs = model.neuron_intervention_experiment(intervention)
    print('more probable candidate per layer, across all neurons in the layer')
    print('candidate1:', intervention.candidates[0], c1)
    print('candidate2:', intervention.candidates[1], c2)

if __name__ == "__main__":
    main()
