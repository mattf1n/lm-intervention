
import torch
# import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import random
from functools import partial
from tqdm import tqdm
# from tqdm import tqdm_notebook

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
                 candidates: list):
        super()
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
                    .repeat(1, 1)


class Model():
    '''
    Wrapper for all model logic
    '''
    def __init__(self):
        super()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.eval()

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
        logits, past = self.model(context)
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

    def neuron_intervention_experiment(self, intervention):
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
