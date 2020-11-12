
from datetime import datetime
import os
import sys
import random

from utils_num_agreement import convert_results_to_pd
from experiment_num_agreement import Intervention, Model
from transformers import GPT2Tokenizer
from vocab_utils import get_nouns, get_nouns2, get_verbs, get_verbs2, get_prepositions, \
        get_preposition_nouns, get_adv1s, get_adv2s
import vocab_utils as vocab

'''
Run all the extraction for a model across many templates
'''

def get_intervention_types():
    return ['indirect', 'direct']

def construct_templates():
    templates = []
    if attractor in  ['singular', 'plural']:
        for p in get_prepositions():
            for ppns, ppnp in get_preposition_nouns():
                ppn = ppns if attractor == 'singular' else ppnp
                template = ' '.join(['The', '{}', p, 'the', ppn])
                templates.append(template)
    elif attractor in ('rc_singular', 'rc_plural', 'rc_singular_no_that', 'rc_plural_no_that'):
        for noun2s, noun2p in get_nouns2():
            noun2 = noun2s if attractor.startswith('rc_singular') else noun2p
            for verb2s, verb2p in get_verbs2():
                verb2 = verb2s if attractor.startswith('rc_singular') else verb2p
                if attractor.endswith('no_that'):
                    template = ' '.join(['The', '{}', 'the', noun2, verb2])
                else:
                    template = ' '.join(['The', '{}', 'that', 'the', noun2, verb2])
                templates.append(template)
    elif attractor in ('within_rc_singular', 'within_rc_plural', 'within_rc_singular_no_that', 'within_rc_plural_no_that'):
        for ns, np in vocab.get_nouns():
            noun = ns if attractor.startswith('within_rc_singular') else np
            if attractor.endswith('no_that'):
                template = ' '.join(['The', noun, 'the', '{}'])
            else:
                template = ' '.join(['The', noun, 'that', 'the', '{}'])
            templates.append(template)
    elif attractor == 'distractor':
        for  adv1 in  get_adv1s():
            for adv2 in get_adv2s():
                templates.append(' '.join(['The', '{}', adv1, 'and', adv2]))
    elif attractor == 'distractor_1':
        for adv1 in get_adv1s():
            templates.append(' '.join(['The', '{}', adv1]))

    else:
        templates = ['The {}']
    return templates

def construct_interventions(tokenizer, DEVICE, attractor, seed, examples):
    interventions = {}
    all_word_count = 0
    used_word_count = 0
    templates = construct_templates()
    for temp in templates:
        if attractor.startswith('within_rc'):
            for noun2s, noun2p in get_nouns2():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try:
                        intervention_name = '_'.join([temp, noun2s, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            [noun2s, noun2p],
                            [v_singular, v_plural],
                            device=DEVICE)
                        used_word_count += 1
                    except Exception as e:
                        pass
        else:
            for ns, np in vocab.get_nouns():
                for v_singular, v_plural in vocab.get_verbs():
                    all_word_count += 1
                    try: 
                        intervention_name = '_'.join([temp, ns, v_singular])
                        interventions[intervention_name] = Intervention(
                            tokenizer,
                            temp,
                            [ns, np],
                            [v_singular, v_plural],
                            device=DEVICE)
                        used_word_count += 1
                    except Exception as e:
                        pass
    print(f"\t Only used {used_word_count}/{all_word_count} nouns due to tokenizer")
    if examples > 0 and len(interventions) >= examples:
        random.seed(seed)
        interventions = {k: v 
                for k, v in random.sample(interventions.items(), examples)}
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
    interventions = construct_interventions(tokenizer, device, attractor, seed,
            examples)
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
    if not (len(sys.argv) == 8):
        print("USAGE: python ", sys.argv[0], 
"<model> <device> <out_dir> <random_weights> <attractor> <seed> <examples>")
    model = sys.argv[1] # distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl
    device = sys.argv[2] # cpu vs cuda
    out_dir = sys.argv[3] # dir to write results
    random_weights = sys.argv[4] == 'random' # true or false
    attractor = sys.argv[5] # singular, plural or none
    seed = int(sys.argv[6]) # to allow consistent sampling
    examples = int(sys.argv[7]) # number of examples to try, 0 for all 
        
    run_all(model, device, out_dir, random_weights, attractor, seed, examples)
