import torch
from transformers import GPT2Tokenizer, TransfoXLTokenizer
from experiment import Intervention, Model
from functools import partial

print('labas!')


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


tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = Model(gpt2_version='transfo-xl-wt103', device='cpu')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = Model(gpt2_version='gpt2', device='cpu')

intervention = Intervention(tokenizer,
                            'The {} said that',
                            ['nurse', 'man', 'woman'],
                            ['he', 'she'])

with torch.no_grad():
    base_representations = get_representations(
        intervention.base_strings_tok[0],
        intervention.position)
    man_representations = get_representations(
        intervention.base_strings_tok[1],
        intervention.position)
    woman_representations = get_representations(
        intervention.base_strings_tok[2],
        intervention.position)
