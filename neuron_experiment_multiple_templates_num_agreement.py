
from datetime import datetime
import os
import sys
import random

from utils_num_agreement import convert_results_to_pd
from experiment_num_agreement import Intervention, Model
from transformers import GPT2Tokenizer
from vocab_utils import get_nouns, get_verbs, get_prepositions, get_preposition_nouns

'''
Run all the extraction for a model across many templates
'''

def get_intervention_types():
    return ['indirect', 'direct']

def construct_pairs(attractor, seed, examples):
    pairs = []
    if attractor in ['singular', 'plural']:
        for ns, np in get_nouns():
            for p in get_prepositions():
                for ppns, ppnp in get_preposition_nouns():
                    ppn = ppns if attractor == 'singular' else ppnp
                    pp = ' '.join([p,'the',ppn])
                    pairs.append(
                            (' '.join(['The',ns, pp]),' '.join(['The',np,pp])))
    else:
        pairs = [('The ' + s, 'The ' + p) for s, p in get_nouns()]
    random.seed(seed)
    return random.sample(pairs, examples)


def construct_interventions(tokenizer, DEVICE, attractor, seed, examples):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    pairs = construct_pairs(attractor, seed, examples)
    print(pairs[0])
    for base, alt in pairs:
        for v_singular, v_plural in get_verbs():
            all_word_count += 1
            try: 
                interventions[base + ' ' + v_singular] = Intervention(
                    tokenizer,
                    '{}',
                    [base, alt],
                    [v_singular, v_plural],
                    device=DEVICE)
                used_word_count += 1
            except:
                pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    return interventions

def run_all(model_type="gpt2", device="cuda", out_dir=".",
        random_weights=False, attractor=None, seed=5, examples=100):
    print("Model:", model_type)
    # Set up all the potential combinations
    intervention_types = get_intervention_types()
    # Initialize Model and Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_type)
    model = Model(device=device, gpt2_version=model_type, 
            random_weights=random_weights)

    # Set up folder if it does not exist
    dt_string = datetime.now().strftime("%Y%m%d")
    folder_name = dt_string+"_neuron_intervention"
    base_path = os.path.join(out_dir, "results", folder_name)
    if random_weights:
        base_path = os.path.join(base_path, "random")
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Fill in all professions into current template
    interventions = construct_interventions(tokenizer, device, 
            attractor, seed, examples)
    # Consider all the intervention types
    for itype in intervention_types:
        print("\t Running with intervention: {}".format(
            itype))
        # Run actual exp
        intervention_results = model.neuron_intervention_experiment(
            interventions, itype, alpha=1.0)

        df = convert_results_to_pd(interventions, intervention_results)
        # Generate file name
        random = ['random'] if random_weights else []
        fcomponents = random + [str(attractor), itype, model_type]
        fname = "_".join(fcomponents)
        # Finally, save each exp separately
        df.to_csv(os.path.join(base_path, fname+".csv"))


if __name__ == "__main__":
    if not (len(sys.argv) == 4 or len(sys.argv) == 5 or len(sys.argv) == 6):
        print("USAGE: python ", sys.argv[0], "<model> <device> <out_dir> [<random_weights>] attractor")
    model = sys.argv[1] # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
    device = sys.argv[2] # cpu vs cuda
    out_dir = sys.argv[3] # dir to write results
    random_weights = sys.argv[4] == 'true' # true or false
    attractor = sys.argv[5] # singular, plural or none
    seed = int(sys.argv[6]) # to allow consistent sampling
    examples = int(sys.argv[7]) # number of examples to try 
        
    run_all(model, device, out_dir, random_weights, attractor, seed, examples)
