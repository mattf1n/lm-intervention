from transformers import GPT2Tokenizer, TransfoXLTokenizer
from experiment import Intervention, Model
from attention_utils import perform_interventions
from pandas import DataFrame
import json


tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
model = Model(gpt2_version='transfo-xl-wt103', output_attentions=True)

# Test experiment
interventions = [
    Intervention(
        tokenizer,
        "The doctor asked the nurse a question. {}",
        ["He", "She"],
        ["asked", "answered"])
]

results = perform_interventions(interventions, model)
