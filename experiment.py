
import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
from functools import partial
from tqdm import tqdm
# from tqdm import tqdm_notebook
import math
import statistics

# from collections import Counter, defaultdict

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from utils import batch, convert_results_to_pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from gpt2_attention import AttentionOverride

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
        self.base_strings_tok = [self.enc.encode(s)
                                 for s in self.base_strings]
        # print(self.base_strings_tok)
        self.base_strings_tok = torch.LongTensor(self.base_strings_tok)\
                                     .to(device)
        # Where to intervene
        self.position = base_string.split().index('{}')

        self.candidates = []
        for c in candidates:
            # '. ' added to input so that tokenizer understand that first word follows a space.
            tokens = self.enc.tokenize('. ' + c)[1:]
            self.candidates.append(tokens)

        self.candidates_tok = [self.enc.convert_tokens_to_ids(tokens)
                               for tokens in self.candidates]


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self,
                 device='cpu',
                 output_attentions=False,
                 gpt2_version='gpt2'):
        super()
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(
            gpt2_version,
            output_attentions=output_attentions)
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

    def get_representations(self, context, position):
        # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
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

    def get_probabilities_for_examples(self, context, candidates):
        """Return probabilities of single-token candidates given context"""
        for c in candidates:
            if len(c) > 1:
                raise ValueError(f"Multiple tokens not allowed: {c}")
        outputs = [c[0] for c in candidates]
        logits, past = self.model(context)[:2]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[:, outputs].tolist()

    def get_probabilities_for_examples_multitoken(self, context, candidates):
        """
        Return probability of multi-token candidates given context.
        Prob of each candidate is normalized by number of tokens.

        Args:
            context: Tensor of token ids in context
            candidates: list of list of token ids in each candidate

        Returns: list containing probability for each candidate
        """
        # TODO: Combine into single batch
        mean_probs = []
        context = context.tolist()
        for candidate in candidates:
            combined = context + candidate
            # Exclude last token position when predicting next token
            batch = torch.tensor(combined[:-1]).unsqueeze(dim=0)
            # Shape (batch_size, seq_len, vocab_size)
            logits = self.model(batch)[0]
            # Shape (seq_len, vocab_size)
            log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
            context_end_pos = len(context) - 1
            continuation_end_pos = context_end_pos + len(candidate)
            token_log_probs = []
            # TODO: Vectorize this
            # Up to but not including last token position
            for i in range(context_end_pos, continuation_end_pos):
                next_token_id = combined[i+1]
                next_token_log_prob = log_probs[i][next_token_id].item()
                token_log_probs.append(next_token_log_prob)
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)
            mean_probs.append(mean_token_prob)
        return mean_probs

    def neuron_intervention(self,
                            context,
                            outputs,
                            rep,
                            layer,
                            neurons,
                            position,
                            intervention_type='diff',
                            alpha=1.):
        # Hook for changing representation during forward pass
        def intervention_hook(module,
                              input,
                              output,
                              position,
                              neurons,
                              intervention,
                              intervention_type):
            # Get the neurons to intervene on
            gather_neurons = [[n] for n in neurons]
            gather_neurons = torch.LongTensor(gather_neurons).to(self.device)
            # First grab the position across batch
            # Then, for each element, get correct index w/ gather
            base = output[:, position, :].gather(
                1, gather_neurons)
            intervention_view = intervention.view_as(base)

            if intervention_type == 'replace':
                base = intervention_view
            elif intervention_type == 'diff':
                base += intervention_view
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")
            # Overwrite values in the output
            # First define mask where to overwrite
            scatter_mask = torch.zeros_like(output).byte()
            for i, v in enumerate(neurons):
                scatter_mask[i, position, v] = 1
            # Then take values from base and scatter
            output.masked_scatter_(scatter_mask, base.flatten())

        intervention_rep = alpha * rep[layer][neurons]
        # Set up the context as batch
        batch_size = len(neurons)
        context = context.unsqueeze(0).repeat(batch_size, 1)
        mlp_intervention_handle = self.model.transformer.h[layer]\
                                      .mlp.register_forward_hook(
            partial(intervention_hook,
                    position=position,
                    neurons=neurons,
                    intervention=intervention_rep,
                    intervention_type=intervention_type))
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
            outputs: candidate outputs
            layer: layer in which to intervene
            attn_override: values to override the computed attention weights.
                           Shape is [batch_size, num_heads, seq_len, seq_len]
            attn_override_mask: indicates which attention weights to override.
                                Shape is [batch_size, num_heads, seq_len, seq_len]
        """

        def intervention_hook(module, input, outputs):
            attention_override_module = AttentionOverride(
                module, attn_override, attn_override_mask)
            outputs[:] = attention_override_module(*input)

        with torch.no_grad():
            hook = self.model.transformer.h[layer].attn.register_forward_hook(
                intervention_hook)
            new_probabilities = self.get_probabilities_for_examples_multitoken(
                context,
                outputs)
            hook.remove()
            return new_probabilities

    def neuron_intervention_experiment(self,
                                       word2intervention,
                                       intervention_type,
                                       alpha=1):
        """
        run multiple intervention experiments
        """

        word2intervention_results = {}
        for word in tqdm(word2intervention, desc='words'):
            word2intervention_results[word] = self.neuron_intervention_single_experiment(
                word2intervention[word], intervention_type, alpha)

        return word2intervention_results

    def neuron_intervention_single_experiment(self,
                                              intervention,
                                              intervention_type,
                                              alpha=100,
                                              bsize=400):
        """
        run one full neuron intervention experiment
        """

        with torch.no_grad():
            '''
            Compute representations for base terms (one for each side of bias)
            '''
            base_representations = self.get_representations(
                intervention.base_strings_tok[0],
                intervention.position)
            man_representations = self.get_representations(
                intervention.base_strings_tok[1],
                intervention.position)
            woman_representations = self.get_representations(
                intervention.base_strings_tok[2],
                intervention.position)

            # TODO: this whole logic can probably be improved
            # determine effect type and set representations

            # e.g. The teacher said that
            if intervention_type == 'man_minus_woman':
                context = intervention.base_strings_tok[0]
                rep = {k: v - woman_representations[k]
                       for k, v in man_representations.items()}
                replace_or_diff = 'diff'
            # e.g. The teacher said that
            elif intervention_type == 'woman_minus_man':
                context = intervention.base_strings_tok[0]
                rep = {k: v - man_representations[k]
                       for k, v in woman_representations.items()}
                replace_or_diff = 'diff'
            # e.g. The man said that
            elif intervention_type == 'man_direct':
                context = intervention.base_strings_tok[1]
                rep = base_representations
                replace_or_diff = 'replace'
            # e.g. The teacher said that
            elif intervention_type == 'man_indirect':
                context = intervention.base_strings_tok[0]
                rep = man_representations
                replace_or_diff = 'replace'
            # e.g. The woman said that
            elif intervention_type == 'woman_direct':
                context = intervention.base_strings_tok[2]
                rep = base_representations
                replace_or_diff = 'replace'
            # e.g. The teacher said that
            elif intervention_type == 'woman_indirect':
                context = intervention.base_strings_tok[0]
                rep = woman_representations
                replace_or_diff = 'replace'
            else:
                raise ValueError(f"Invalid intervention_type: {intervention_type}")

            # Probabilities without intervention (Base case)
            candidate1_base_prob, candidate2_base_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[0].unsqueeze(0),
                intervention.candidates_tok)[0]
            candidate1_alt1_prob, candidate2_alt1_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[1].unsqueeze(0),
                intervention.candidates_tok)[0]
            candidate1_alt2_prob, candidate2_alt2_prob = self.get_probabilities_for_examples(
                intervention.base_strings_tok[2].unsqueeze(0),
                intervention.candidates_tok)[0]
            # print("base", candidate1_base_prob, candidate2_base_prob)
            # print("man", candidate1_alt1_prob, candidate2_alt1_prob)
            # print("woman", candidate1_alt2_prob, candidate2_alt2_prob)

            # Now intervening on potentially biased example

            candidate1_probs = torch.zeros((self.num_layers, self.num_neurons))
            candidate2_probs = torch.zeros((self.num_layers, self.num_neurons))
            # Intervene at every possible neuron
            for layer in range(self.num_layers):
                for neurons in batch(range(self.num_neurons), bsize):
                    probs = self.neuron_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        rep=rep,
                        layer=layer,
                        neurons=neurons,
                        position=intervention.position,
                        intervention_type=replace_or_diff,
                        alpha=alpha)
                    for neuron, (p1, p2) in zip(neurons, probs):
                        candidate1_probs[layer][neuron] = p1
                        candidate2_probs[layer][neuron] = p2
                    break
                break

        return (candidate1_base_prob, candidate2_base_prob,
                candidate1_alt1_prob, candidate2_alt1_prob,
                candidate1_alt2_prob, candidate2_alt2_prob,
                candidate1_probs, candidate2_probs)

    def attention_intervention_experiment(self, intervention, effect):
        """
        Run one full attention intervention experiment
        measuring indirect or direct effect.
        """
        # E.g. The doctor asked the nurse a question. He
        x = intervention.base_strings_tok[0]
        # E.g. The doctor asked the nurse a question. She
        x_alt = intervention.base_strings_tok[1]

        if effect == 'indirect':
            input = x_alt  # Get attention for x_alt
        elif effect == 'direct':
            input = x  # Get attention for x
        else:
            raise ValueError(f"Invalid effect: {effect}")
        attention_override = self.model(self._tok_to_batch(input))[-1]

        batch_size = 1
        seq_len = len(x)
        seq_len_alt = len(x_alt)
        assert seq_len == seq_len_alt
        assert len(attention_override) == self.num_layers
        assert attention_override[0].shape == (batch_size, self.num_heads, seq_len, seq_len)

        with torch.no_grad():

            candidate1_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate2_probs_head = torch.zeros((self.num_layers, self.num_heads))
            candidate1_probs_layer = torch.zeros(self.num_layers)
            candidate2_probs_layer = torch.zeros(self.num_layers)

            if effect == 'indirect':
                context = x
            else:
                context = x_alt

            # Intervene at every layer and head by overlaying attention induced by x_alt
            for layer in range(self.num_layers):
                layer_attention_override = attention_override[layer]
                attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8)
                candidate1_probs_layer[layer], candidate2_probs_layer[layer] = self.attention_intervention(
                    context=context,
                    outputs=intervention.candidates_tok,
                    layer=layer,
                    attn_override=layer_attention_override,
                    attn_override_mask=attention_override_mask)
                for head in range(self.num_heads):
                    attention_override_mask = torch.zeros_like(layer_attention_override, dtype=torch.uint8)
                    attention_override_mask[0][head] = 1 # Set mask to 1 for single head only
                    candidate1_probs_head[layer][head], candidate2_probs_head[layer][head] = self.attention_intervention(
                        context=context,
                        outputs=intervention.candidates_tok,
                        layer=layer,
                        attn_override=layer_attention_override,
                        attn_override_mask=attention_override_mask)

        return candidate1_probs_head, candidate2_probs_head, candidate1_probs_layer, candidate2_probs_layer

    def _tok_to_batch(self, tok_ids):
        return torch.tensor(tok_ids).unsqueeze(0)


def main():
    DEVICE = 'cpu'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = Model(device=DEVICE)

    base_sentence = "The {} said that"
    biased_word = "teacher"
    intervention = Intervention(
            tokenizer,
            base_sentence,
            [biased_word, "man", "woman"],
            ["he", "she"],
            device=DEVICE)
    interventions = {biased_word: intervention}

    intervention_results = model.neuron_intervention_experiment(
        interventions, 'man_minus_woman')
    df = convert_results_to_pd(
        interventions, intervention_results)
    print('more probable candidate per layer, across all neurons in the layer')
    print(df[0:5])


if __name__ == "__main__":
    main()
