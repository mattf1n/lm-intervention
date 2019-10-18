# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
import os
import sys

from datetime import datetime
from experiment import Intervention, Model
from utils import convert_results_to_pd
from transformers import GPT2Tokenizer

# sns.set(style="ticks", color_codes=True)

DEVICE = 'cpu'


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = Model(DEVICE)


# Currently not implemented/needed functions.
def plot_log_probs(layer_to_candidate1_log_probs, layer_to_candidate2_log_probs):

    raise NotImplementedError


def print_neuron_hook(module, input, output, position, neuron):
    # print(output.shape)
    print(output[0][position][neuron])


def print_all_hook(module, input, output, position):
    # print(output.shape)
    print(output[0][position])


# ### Bookkeeping for all the experiments

# note: does not include teacher
interventions = {}
with open('professions_neutral.json', 'r') as f:
    all_word_count = 0
    used_word_count = 0
    for l in f:
        # there is only one line that eval's to an array
        for j in eval(l):
            all_word_count += 1
            biased_word = j[0]
            try:
                interventions[biased_word] = Intervention(
                        tokenizer,
                        "The {} said that",
                        [biased_word, "man", "woman"],
                        ["he", "she"],
                        device=DEVICE)
                used_word_count += 1
            except:
                pass
                # print("excepted {} due to tokenizer splitting.".format(
                #     biased_word))
    print("Only used {}/{} neutral words due to tokenizer".format(
        used_word_count, all_word_count))


# # ToDo: Need to double check that the sentences all make sense
# male_interventions = []
# with open('male_word_said.txt', 'r') as f:
#     all_word_count = 0
#     used_word_count = 0
#     for l in f:
#         # Strip off the \n
#         biased_word = l[:-1]
#         all_word_count += 1
#         try:
#             male_interventions.append(
#                     Intervention(
#                         tokenizer,
#                         "The {} said that",
#                         [biased_word, "man", "woman"],
#                         ["he", "she"]))
#             used_word_count += 1
#         except:
#             pass
#             # print("excepted {} due to tokenizer splitting.".format(
#             #     biased_word))
#     print("Only used {}/{} male words due to tokenizer".format(
#         used_word_count, all_word_count))


# # ToDo: Need to double check that the sentences all make sense
# female_interventions = []
# with open('female_word_said.txt', 'r') as f:
#     all_word_count = 0
#     used_word_count = 0
#     for l in f:
#         # Strip off the \n
#         biased_word = l[:-1]
#         all_word_count += 1
#         try:
#             female_interventions.append(
#                     Intervention(
#                         tokenizer,
#                         "The {} said that",
#                         [biased_word, "man", "woman"],
#                         ["he", "she"]))
#             used_word_count += 1
#         except:
#             pass
#             # print("excepted {} due to tokenizer splitting.".format(
#             #     biased_word))
#     print("Only used {}/{} female words due to tokenizer".format(
#         used_word_count, all_word_count))


# Multiple Professions Experiment Run
def run(intervention_type, output_csv_file, alpha):
    print(f"intervention_type: {intervention_type}")
    print(f"output_csv_file: {output_csv_file}")
    print(f"alpha: {alpha}")

    intervention_results = model.neuron_intervention_experiment(
        interventions, intervention_type, alpha)

    df = convert_results_to_pd(interventions, intervention_results)
    df.head()
    df.to_csv(output_csv_file)


def run_all():
    '''
    Runs all the interventions and saves in datetime named folder
    '''

    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string+"_neuron_intervention"
    base_path = os.path.join("results", folder_name)
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Initialize intervention types
    intervention_types = ['man_minus_woman',
                          'woman_minus_man',
                          'man_direct',
                          'man_indirect',
                          'woman_direct',
                          'woman_indirect']

    for itype in intervention_types:
        print("Running {}...".format(itype))
        intervention_results = model.neuron_intervention_experiment(
            interventions, itype, alpha=1.0)

        df = convert_results_to_pd(interventions, intervention_results)
        df.head()
        df.to_csv(os.path.join(base_path, itype+".csv"))


if __name__ == '__main__':
    if sys.argv[1] == "all":
        run_all()
    elif len(sys.argv) == 4:
        run(sys.argv[1], sys.argv[2], float(sys.argv[3]))
    else:
        print("python", sys.argv[0], "<intervention_type", "<output_csv_file>", "<alpha>")
