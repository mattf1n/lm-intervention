from datetime import datetime
import os
import sys

from utils import convert_results_to_pd
from experiment import Intervention, Model
from transformers import GPT2Tokenizer, TransfoXLTokenizer

'''
Run all the extraction for a model across many templates
'''


def get_profession_list():
    # Get the list of all considered professions

    word_list = []
    with open('professions.json', 'r') as f:
        for l in f:
            # there is only one line that eval's to an array
            for j in eval(l):
                word_list.append(j[0])
    return word_list


def get_template_list(indices=None):
    # Get list of all considered templates
    # "That" sentences are ours
    # "Because" sentences are a subset
    # from https://arxiv.org/pdf/1807.11714.pdf (Lu et al.)
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
    if indices:
        subset_templates = [templates[i-1] for i in indices]
        print("subset of templates:", subset_templates)
        return subset_templates

    return templates


def get_intervention_types():
    return [#'man_minus_woman',
            #'woman_minus_man',
            'man_direct',
            'man_indirect',
            'woman_direct',
            'woman_indirect']


def construct_interventions(base_sent, professions, tokenizer, DEVICE):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    for p in professions:
        all_word_count += 1
        try:
            interventions[p] = Intervention(
                tokenizer,
                base_sent,
                [p, "man", "woman"],
                ["he", "she"],
                device=DEVICE)
            used_word_count += 1
        except:
            pass
    print("\t Only used {}/{} professions due to tokenizer".format(
        used_word_count, all_word_count))
    return interventions


def run_all(model_type="gpt2", device="cuda", out_dir=".", random_weights=False, template_indices=None):
    print("Model:", model_type, flush=True)
    # Set up all the potential combinations
    professions = get_profession_list()
    templates = get_template_list(template_indices)
    intervention_types = get_intervention_types()
    # Initialize Model and Tokenizer
    model = Model(device=device, gpt2_version=model_type, random_weights=random_weights)
    ### New ###
    if model.is_txl:
        tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_version)
    ### New ###

    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string+"_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Iterate over all possible templates
    for temp in templates:
        print("Running template \"{}\" now...".format(
            temp), flush=True)
        # Fill in all professions into current template
        interventions = construct_interventions(
            temp, professions, tokenizer, device)
        # Consider all the intervention types
        for itype in intervention_types:
            print("\t Running with intervention: {}".format(
                itype), flush=True)
            # Run actual exp
            intervention_results = model.neuron_intervention_experiment(
                interventions, itype, alpha=1.0)

            df = convert_results_to_pd(interventions, intervention_results)
            # Generate file name
            temp_string = "_".join(temp.replace("{}", "X").split())
            model_type_string = model_type
            fname = "_".join([temp_string, itype, model_type_string])
            # Finally, save each exp separately
            df.to_csv(os.path.join(base_path, fname+".csv"))


if __name__ == "__main__":
    if not (len(sys.argv) == 4 or len(sys.argv) == 5 or len(sys.argv) == 6):
        print("USAGE: python ", sys.argv[0], "<model> <device> <out_dir> [<random_weights>] [<template_indices>]")
        print('<template_indices> is an optional comma-separated list of indices of templates to use, 1-indexed')
    model = sys.argv[1] # cpu vs cuda
    device = sys.argv[2] # gpt2, gpt2-medium, gpt2-large
    out_dir = sys.argv[3] # dir to write results

    ### NEW ###
    # random_weights = False
    # if sys.argv[4] and sys.argv[4] == 'true':
    #     random_weights = True
    #
    # template_indices = None
    # if sys.argv[5]:
    #     template_indices = [int(ind) for ind in sys.argv[5].split(',')]
    random_weights = False
    if len(sys.argv) >= 5 and sys.argv[4] == 'true':
        random_weights = True

    template_indices = None
    if len(sys.argv) >= 6 and sys.argv[5]:
        template_indices = [int(ind) for ind in sys.argv[5].split(',')]
    ### NEW ###

    run_all(model, device, out_dir, random_weights=random_weights, template_indices=template_indices)
