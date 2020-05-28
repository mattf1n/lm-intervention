import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, BertTokenizer
from experiment import Intervention, Model
from txl_attention import TXLAttentionOverride
from functools import partial
import math
import statistics
import sys


# def txl_intervention_hook(module, input, outputs, attn_override, attn_override_mask):
#     attention_override_module = TXLAttentionOverride(
#         module, attn_override, attn_override_mask)
#
#     batch_size = attn_override.shape[0]
#     q_len = input[0].shape[0]
#     mem_len = model.model.transformer.mem_len
#     mems = torch.zeros((mem_len, batch_size, model.model.transformer.d_model), device=model.device)
#     all_ones = torch.ones((q_len, mem_len + q_len), dtype=torch.uint8, device=model.device)
#     attn_mask = (torch.triu(all_ones, 1 + mem_len) + torch.tril(all_ones, 0))[:, :, None]
#
#     outputs[:] = attention_override_module(*input, attn_mask=attn_mask, mems=mems)
#
#
#
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = Model(output_attentions=True, gpt2_version='bert-base-uncased')

intervention = Intervention(tokenizer,
                            'The guard saved the editor from the criminals because {}',
                            ['she', 'he'],
                            ['needed help.', 'was on duty.'])


x = intervention.base_strings_tok[0]
x_alt = intervention.base_strings_tok[1]

# print(intervention.base_strings)
# print(x)
# print(x_alt)
# print(intervention.candidates)
# print(intervention.candidates_tok)
# assert False

batch = x_alt.clone().detach().unsqueeze(0).to(model.device)
attention_override = model.model(batch)[-1]

def get_probabilities_for_examples_multitoken_bert(context, candidates, regime=1):
    """
    Return probability of multi-token candidates given context.
    Prob of each candidate is normalized by number of tokens.

    Args:
        context: Tensor of token ids in context
        candidates: list of list of token ids in each candidate

    Returns: list containing probability for each candidate
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    mask_id = tokenizer.convert_tokens_to_ids('[MASK]')

    # TODO: Combine into single batch
    mean_probs = []
    context = context.tolist()
    for candidate in candidates:
        print('\nCandidate:', tokenizer.decode(candidate))
        token_log_probs = []
        for i, c in enumerate(candidate):
            combined = context + candidate[:i] + [mask_id]
            if regime == 2: combined = combined + candidate[i+1:]
            if regime == 3: combined = combined + [mask_id] * len(candidate[i+1:])
            print(tokenizer.decode(combined))
            pred_idx = len(context) + i
            batch = torch.tensor(combined).unsqueeze(dim=0).to(model.model.device)
            # Shape (batch_size, seq_len, vocab_size)
            logits = model.model(batch)[0]
            # Shape (seq_len, vocab_size)
            log_probs = F.log_softmax(logits[-1, :, :], dim=-1)
            next_token_log_prob = log_probs[pred_idx][c].item()
            token_log_probs.append(next_token_log_prob)
        mean_token_log_prob = statistics.mean(token_log_probs)
        mean_token_prob = math.exp(mean_token_log_prob)
        mean_probs.append(mean_token_prob)
    return mean_probs

for regime in [1, 2, 3]:
    print(f'\n****** Regime ({regime}) ******')
    new_probabilities = get_probabilities_for_examples_multitoken_bert(
        x,
        intervention.candidates_tok,
        regime)

    print('\nProbabilities for candidates:', new_probabilities)

# with torch.no_grad():
#     layer = 8
#     layer_attention_override = attention_override[layer]
#     attention_override_mask = torch.ones_like(layer_attention_override, dtype=torch.uint8)
#
#     hook = model.model.transformer.layers[layer].dec_attn.register_forward_hook(
#         partial(txl_intervention_hook,
#                 attn_override=layer_attention_override,
#                 attn_override_mask=attention_override_mask))
#
#     new_probabilities = model.get_probabilities_for_examples_multitoken(
#         x,
#         intervention.candidates_tok)
#
#     hook.remove()
#     print(new_probabilities)
