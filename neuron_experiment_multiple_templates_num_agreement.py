from datetime import datetime
import os
import sys

from utils_num_agreement import convert_results_to_pd
from experiment_num_agreement import Intervention, Model
from transformers import GPT2Tokenizer
from vocab_utils import get_nouns, get_verbs, get_prepositions, get_preposition_nouns

'''
Run all the extraction for a model across many templates
'''


def get_profession_list():
    # Get the list of all considered professions

    word_list = []
    with open('singular_nouns.json', 'r') as f:
        for l in f:
            # there is only one line that eval's to an array
            for j in eval(l):
                word_list.append(j)
    return word_list

def get_template_list():
    return ['The {}']

def get_intervention_types():
    return [#'man_minus_woman',
            #'woman_minus_man',
            #'man_direct',
            #'man_indirect',
            #'woman_direct:,
            # 'woman_indirect'
            'indirect',
            'direct']

def construct_interventions(base_sent, tokenizer, DEVICE, attractor=None):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    alts = get_nouns()
    if attractor:
        alts = []
        for ns, np in get_nouns():
            for p in get_prepositions():
                for ppns, ppnp in get_preposition_nouns():
                    ppn = ppns if attractor == 'singular' else ppnp
                    pp = ' '.join([p,'the',ppn])
                    alts.append((' '.join([ns, pp]),' '.join([np,pp])))

    for base, alt in alts:
        all_word_count += 1
        for v_singular, v_plural in get_verbs():
            try: 
                interventions[base + ' ' + v_singular] = Intervention(
                    tokenizer,
                    base_sent,
                    [base, alt],
                    [v_singular, v_plural],
                    device=DEVICE)
                used_word_count += 1
            except:
                pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    return interventions

def run_all(model_type="gpt2", device="cuda", out_dir=".",
        random_weights=False, attractor=None):
    print("Model:", model_type)
    # Set up all the potential combinations
    templates = get_template_list()
    intervention_types = get_intervention_types()
    # Initialize Model and Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, random_weights=random_weights)

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
            temp))
        # Fill in all professions into current template
        interventions = construct_interventions(
            temp, tokenizer, device, attractor=attractor)
        # Consider all the intervention types
        for itype in intervention_types:
            print("\t Running with intervention: {}".format(
                itype))
            # Run actual exp
            intervention_results = model.neuron_intervention_experiment(
                interventions, itype, alpha=1.0)

            df = convert_results_to_pd(interventions, intervention_results)
            # Generate file name
            # temp_string = "_".join(temp.replace("{}", "X").split())
            random = ['random'] if random_weights else []
            fcomponents = random + [str(attractor), itype, model_type]
            fname = "_".join(fcomponents)
            # Finally, save each exp separately
            df.to_csv(os.path.join(base_path, fname+".csv"))


if __name__ == "__main__":
    if not (len(sys.argv) == 4 or len(sys.argv) == 5 or len(sys.argv) == 6):
        print("USAGE: python ", sys.argv[0], "<model> <device> <out_dir> [<random_weights>] attractor")
        print('<template_indices> is an optional comma-separated list of indices of templates to use, 1-indexed')
    model = sys.argv[1] # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
    device = sys.argv[2] # cpu vs cuda
    out_dir = sys.argv[3] # dir to write results
    attractor = None if sys.argv[5] == 'None' else sys.argv[5]

    random_weights = False
    if sys.argv[4] and sys.argv[4] == 'true':
        random_weights = True
        
    run_all(model, device, out_dir, random_weights=random_weights, attractor=attractor)
