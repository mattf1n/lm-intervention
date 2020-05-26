import torch
from transformers import GPT2Tokenizer, TransfoXLTokenizer
from experiment import Intervention, Model
from txl_attention import TXLAttentionOverride
from functools import partial



def txl_intervention_hook(module, input, outputs, attn_override, attn_override_mask):
    attention_override_module = TXLAttentionOverride(
        module, attn_override, attn_override_mask)

    batch_size = attn_override.shape[0]
    q_len = input[0].shape[0]
    mem_len = model.model.transformer.mem_len
    mems = torch.zeros((mem_len, batch_size, model.model.transformer.d_model), device=model.device)
    all_ones = torch.ones((q_len, mem_len + q_len), dtype=torch.uint8, device=model.device)
    attn_mask = (torch.triu(all_ones, 1 + mem_len) + torch.tril(all_ones, 0))[:, :, None]

    outputs[:] = attention_override_module(*input, attn_mask=attn_mask, mems=mems)



tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = Model(output_attentions=True, gpt2_version='transfo-xl-wt103')

# intervention = Intervention(tokenizer,
#                             "The doctor asked the nurse a question. {}",
#                             ["He", "She"],
#                             ["asked", "answered"])
# intervention = Intervention(tokenizer,
#                             'The cook prepared a dish for the teacher because {}',
#                             ['She', 'He'],
#                             ['is hungry', 'just learned a new dish.'])
intervention = Intervention(tokenizer,
                            'The guard saved the editor from the criminals because {}',
                            ['she', 'he'],
                            ['needed help.', 'was on duty.'])


x = intervention.base_strings_tok[0]
x_alt = intervention.base_strings_tok[1]

# print(x)
# print(x_alt)
# print(intervention.candidates)
# print(intervention.candidates_tok)
# assert False

batch = x_alt.clone().detach().unsqueeze(0).to(model.device)
attention_override = model.model(batch)[-1]


with torch.no_grad():
    layer = 8
    layer_attention_override = attention_override[layer]
    attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8)

    hook = model.model.transformer.layers[layer].dec_attn.register_forward_hook(
        partial(txl_intervention_hook,
                attn_override=layer_attention_override,
                attn_override_mask=attention_override_mask))

    new_probabilities = model.get_probabilities_for_examples_multitoken(
        x,
        intervention.candidates_tok)

    hook.remove()
    print(new_probabilities)
