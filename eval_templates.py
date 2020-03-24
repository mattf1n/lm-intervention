from experiment import Model, Intervention
from transformers import GPT2Tokenizer
import sys
import warnings
import numpy as np

import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
model_version = 'gpt2'
m = Model(gpt2_version=model_version)

templates = ["The {} said that",
             "The {} yelled that",
             "The {} whispered that",
             "The {} wanted that",
             "The {} desired that",
             "The {} wished that",
             "The {} ate because",
             "The {} ran because",
             "The {} drove because",
             "The {} slept because",
             "The {} cried because",
             "The {} laughed because",
             "The {} went home because",
             "The {} stayed up because",
             "The {} was fired because",
             "The {} was promoted because",
             "The {} yelled because"]

substitutes = ['woman', 'man']
candidates = ['she', 'he']

man_he_she_vals = []
woman_she_he_vals = []

tokenizer = GPT2Tokenizer.from_pretrained(model_version)
for template in templates:
    intervention = Intervention(
        tokenizer,
        base_string=template,
        substitutes=substitutes,
        candidates=candidates
    )
    for i, context in enumerate(intervention.base_strings_tok):
        context = context.unsqueeze(0).repeat(1, 1)
        base = template.format(substitutes[i])
        probs = m.get_probabilities_for_examples(context, intervention.candidates_tok)[0]
        she_prob, he_prob = probs
        stereo = probs[i]
        anti = probs[i-1]
        print(f"{base} | she: {she_prob:.4f} | he: {he_prob: .4f} | she/he: {she_prob / he_prob: .4f} | anti/steroe: {anti / stereo: .4f}")
        if i == 0:
            woman_she_he_vals.append(she_prob / he_prob)
        else:
            man_he_she_vals.append(he_prob / she_prob)

print("man:")
print("mean:", np.mean(man_he_she_vals))
print("min:", min(man_he_she_vals))
print("max:", max(man_he_she_vals))
print("std:", np.std(man_he_she_vals))

print("woman:")
print("min:", min(woman_she_he_vals))
print("max:", max(woman_she_he_vals))
print("mean:", np.mean(woman_she_he_vals))
print("std:", np.std(woman_she_he_vals))



