import torch
from transformers import GPT2Tokenizer, TransfoXLTokenizer
from experiment import Intervention, Model
from functools import partial
from utils import batch
import numpy as np



def get_representations(context, position):
    # Hook for saving the representation
    def extract_representation_hook(module,
                                    input,
                                    output,
                                    position,
                                    representations,
                                    layer):
        # output shape: [seq_len, batch_size, hidden_dim]
        representations[layer] = output[position][0]
    handles = []
    representation = {}
    with torch.no_grad():
        # construct all the hooks
        # word embeddings will be layer -1
        handles.append(model.model.transformer.word_emb.register_forward_hook(
            partial(extract_representation_hook,
                    position=position,
                    representations=representation,
                    layer=-1)))
        # hidden layers
        for layer in range(model.num_layers):
            handles.append(model.model.transformer.layers[layer]\
                               .pos_ff.register_forward_hook(
                partial(extract_representation_hook,
                        position=position,
                        representations=representation,
                        layer=layer)))
        logits, past = model.model(context.unsqueeze(0))
        for h in handles:
            h.remove()
    # print(representation[0][:5])
    return representation


def txl_intervention_hook(module,
                          input,
                          output,
                          position,
                          neurons,
                          intervention,
                          intervention_type):
    # Get the neurons to intervene on
    neurons = torch.LongTensor(neurons).to(model.device)
    # First grab the position across batch
    # Then, for each element, get correct index w/ gather
    # output shape: [seq_len, batch_size, hidden_dim] (different than gpt2)
    base = output[position, :, :].gather(1, neurons)
    intervention_view = intervention.view_as(base)

    if intervention_type == 'replace':
        base = intervention_view
    elif intervention_type == 'diff':
        base += intervention_view
    else:
        raise ValueError(f"Invalid intervention_type: {intervention_type}")
    # Overwrite values in the output
    # First define mask where to overwrite
    ### NEW ###
    # scatter_mask = torch.zeros_like(output).byte()
    scatter_mask = torch.zeros_like(output, dtype=torch.bool)
    ### NEW ###
    for i, v in enumerate(neurons):
        scatter_mask[position, i, v] = 1
    # Then take values from base and scatter
    output.masked_scatter_(scatter_mask, base.flatten())


def neuron_intervention(context,
                        outputs,
                        rep,
                        layers,
                        neurons,
                        position,
                        intervention_type='diff',
                        alpha=1.):

    # Set up the context as batch
    batch_size = len(neurons)
    context = context.unsqueeze(0).repeat(batch_size, 1)
    handle_list = []
    for layer in set(layers):
        neuron_loc = np.where(np.array(layers) == layer)[0]
        n_list = []
        for n in neurons:
            unsorted_n_list = [n[i] for i in neuron_loc]
            n_list.append(list(np.sort(unsorted_n_list)))
        intervention_rep = alpha * rep[layer][n_list]
        if layer == -1:
            handle_list.append(model.model.transformer.word_emb.register_forward_hook(
                partial(txl_intervention_hook,
                        position=position,
                        neurons=n_list,
                        intervention=intervention_rep,
                        intervention_type=intervention_type)))
        else:
            handle_list.append(model.model.transformer.layers[layer]\
                               .pos_ff.register_forward_hook(
                partial(txl_intervention_hook,
                        position=position,
                        neurons=n_list,
                        intervention=intervention_rep,
                        intervention_type=intervention_type)))
    new_probabilities = model.get_probabilities_for_examples(
        context,
        outputs)
    for hndle in handle_list:
        hndle.remove()
    return new_probabilities


tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = Model(gpt2_version='transfo-xl-wt103', device='cpu')

intervention = Intervention(tokenizer,
                            'The {} said that',
                            ['nurse', 'man', 'woman'],
                            ['he', 'she'])

with torch.no_grad():
    # The nurse said that
    base_representations = get_representations(
        intervention.base_strings_tok[0],
        intervention.position)
    # The man said that
    man_representations = get_representations(
        intervention.base_strings_tok[1],
        intervention.position)
    # The woman said that
    woman_representations = get_representations(
        intervention.base_strings_tok[2],
        intervention.position)

    # Intervention type: man direct
    context = intervention.base_strings_tok[1]
    rep = base_representations
    replace_or_diff = 'replace'

    # Probabilities (for 'he' vs 'she') without intervention (Base case)
    candidate1_base_prob, candidate2_base_prob = model.get_probabilities_for_examples(
        intervention.base_strings_tok[0].unsqueeze(0),
        intervention.candidates_tok)[0]
    candidate1_alt1_prob, candidate2_alt1_prob = model.get_probabilities_for_examples(
        intervention.base_strings_tok[1].unsqueeze(0),
        intervention.candidates_tok)[0]
    candidate1_alt2_prob, candidate2_alt2_prob = model.get_probabilities_for_examples(
        intervention.base_strings_tok[2].unsqueeze(0),
        intervention.candidates_tok)[0]

    # intervention_loc: all
    candidate1_probs = torch.zeros((model.num_layers + 1, model.num_neurons))
    candidate2_probs = torch.zeros((model.num_layers + 1, model.num_neurons))

    bsize = 800
    alpha = 1
    for layer in range(-1, model.num_layers):
        print(f'working on layer {layer}...')
        for neurons in batch(range(model.num_neurons), bsize):
            neurons_to_adj, layers_to_adj = [], []
            neurons_to_search = [[i] + neurons_to_adj for i in neurons]
            layers_to_search = [layer] + layers_to_adj

            probs = neuron_intervention(
                context=context,
                outputs=intervention.candidates_tok,
                rep=rep,
                layers=layers_to_search,
                neurons=neurons_to_search,
                position=intervention.position,
                intervention_type=replace_or_diff,
                alpha=alpha)
            for neuron, (p1, p2) in zip(neurons, probs):
                candidate1_probs[layer + 1][neuron] = p1
                candidate2_probs[layer + 1][neuron] = p2
