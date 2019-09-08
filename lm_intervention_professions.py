import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
from tqdm import tqdm_notebook

from pytorch_transformers import GPT2Tokenizer


from experiment import Intervention, Model
from utils import convert_results_to_pd


DEVICE = 'cuda'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = Model(DEVICE)


# Currently not implemented/needed functions. 
def plot_log_probs(layer_to_candidate1_log_probs, layer_to_candidate2_log_probs):
    
    raise NotImplementedError
        
def print_neuron_hook(module, input, output, position, neuron):
        #print(output.shape) 
        print(output[0][position][neuron])
        
def print_all_hook(module, input, output, position):
    #print(output.shape) 
    print(output[0][position])


# ### Bookkeeping for all the experiments

# note: does not include teacher
interventions = {}
with open('professions_neutral.json', 'r') as f:
    for l in f: 
        # there is only one line that eval's to an array
        for j in eval(l):
            biased_word = j[0]
            interventions[biased_word] = Intervention(
                    tokenizer,
                    "The {} said that",
                    [biased_word, "man", "woman"],
                    ["he", "she"],
                    device=DEVICE)


# ToDo: Need to double check that the sentences all make sense
male_interventions = []
with open('male_word_said.txt', 'r') as f:
    for l in f:
        # Strip off the \n
        biased_word = l[:-1]
        male_interventions.append(
                Intervention(
                    tokenizer,
                    "The {} said that",
                    [biased_word, "man", "woman"],
                    ["he", "she"]))


# ToDo: Need to double check that the sentences all make sense
female_interventions = []
with open('female_word_said.txt', 'r') as f:
    for l in f:
        # Strip off the \n
        biased_word = l[:-1]
        female_interventions.append(
                Intervention(
                    tokenizer,
                    "The {} said that",
                    [biased_word, "man", "woman"],
                    ["he", "she"]))


# ## Multiple Professions Experiment Run


intervention_results = model.neuron_intervention_experiment(interventions)

df = convert_results_to_pd(interventions, intervention_results)
df.head()
df.to_csv("lm_intervention_professions_results_alpha1.pkl")


