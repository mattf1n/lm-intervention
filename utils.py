import pandas as pd


def convert_results_to_pd(interventions, intervention_results):  
    """
    Convert intervention results to data frame

    Args:
        interventions: dictionary from word (e.g., profession) to intervention
        intervention_results: dictionary from word to intervention results
    """  

    results = []
    for word in intervention_results:
        intervention = interventions[word]
        _, c1_probs, _, c2_probs = intervention_results[word] 
        for c1_neurons, c2_neurons in zip(c1_probs.items(), c2_probs.items()):
            cur_layer = c1_neurons[0]
            first_probs = c1_neurons[1]
            second_probs = c2_neurons[1]
            for ix, (p1, p2) in enumerate(zip(first_probs, second_probs)):
                results.append({
                     'word': word,
                     'base_string': intervention.base_strings[0],
                     'first_condition': intervention.candidates[0],
                     'second_condition': intervention.candidates[1],
                     'p1': float(p1),
                     'p2': float(p2),
                     'layer': cur_layer,
                     'neuron': ix})
    return pd.DataFrame(results)