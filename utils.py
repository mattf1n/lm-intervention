import pandas as pd


def batch(iterable, bsize=1):
    total_len = len(iterable)
    for ndx in range(0, total_len, bsize):
        yield list(iterable[ndx:min(ndx + bsize, total_len)])


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
        candidate1_base_prob, candidate2_base_prob,\
            candidate1_alt1_prob, candidate2_alt1_prob,\
            candidate1_alt2_prob, candidate2_alt2_prob,\
            candidate1_probs, candidate2_probs = intervention_results[word]
        for layer in range(candidate1_probs.size(0)):
            for neuron in range(candidate1_probs.size(1)):
                c1_prob, c2_prob = candidate1_probs[layer][neuron], candidate2_probs[layer][neuron]
                results.append({

                    # strings
                    'word': word,
                    'base_string': intervention.base_strings[0],
                    'alt_string1': intervention.base_strings[1],
                    'alt_string2': intervention.base_strings[2],
                    'candidate1': intervention.candidates[0],
                    'candidate2': intervention.candidates[1],

                    # base probs
                    'candidate1_base_prob': float(candidate1_base_prob),
                    'candidate2_base_prob': float(candidate2_base_prob),
                    'candidate1_alt1_prob': float(candidate1_alt1_prob),
                    'candidate2_alt1_prob': float(candidate2_alt1_prob),
                    'candidate1_alt2_prob': float(candidate1_alt2_prob),
                    'candidate2_alt2_prob': float(candidate2_alt2_prob),

                    # intervention probs
                    'candidate1_prob': float(c1_prob),
                    'candidate2_prob': float(c2_prob),
                    'layer': layer,
                    'neuron': neuron})
    return pd.DataFrame(results)
