import unittest

import torch.nn.functional as F
from experiment import Model, Intervention
from attention_utils import perform_intervention
import torch
from transformers import GPT2Tokenizer
import numpy as np
from scipy.stats.mstats import gmean


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(gpt2_version='distilgpt2', output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    def test_get_probabilities_for_examples(self):

        examples = [
            ("The teacher said that", ["she will multiplewordpiece", "he did teach there"]),
            ("The multiplewordpiece said that", ["she is", "he"]),
            ("The teacher said that", ["she", "he"])
        ]
        for context, candidates in examples:
            context_tokens = self.tokenizer.tokenize(context)
            # Compute expected probs recursively by incrementally adding next token and
            # calling self._get_probabilities_for_examples_single_token
            expected_probs = []
            for candidate in candidates:
                probs = []
                candidate_tokens = self.tokenizer.tokenize(candidate)
                candidate_tokens[0] = 'Ġ' + candidate_tokens[0]
                for i in range(len(candidate_tokens)):
                    new_context_tokens = context_tokens + candidate_tokens[:i]
                    new_context_ids = self.tokenizer.convert_tokens_to_ids(new_context_tokens)
                    new_candidate_token = candidate_tokens[i]
                    new_candidate_id = self.tokenizer.convert_tokens_to_ids(new_candidate_token)
                    prob = self._get_probabilities_for_examples_single_token(new_context_ids, [new_candidate_id])[0]
                    probs.append(prob)
                mean_prob = gmean(probs)
                expected_probs.append(mean_prob)

            # Compute actual probs using method being tested
            candidates_ids = []
            for candidate in candidates:
                tokens = self.tokenizer.tokenize(candidate)
                tokens[0] = 'Ġ' + tokens[0]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                candidates_ids.append(token_ids)
            context_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(context_tokens))
            probs = self.model.get_probabilities_for_examples_multitoken(context_ids, candidates_ids)
            np.testing.assert_almost_equal(expected_probs, probs, 5)

    def test_perform_intervention(self):
        # Call attention_intervention_experiment and verify that the the direct and indirect effects are different
        intervention = Intervention(
            self.tokenizer,
            'The doctor asked the nurse a question. {}',
            ['he', 'she'],
            ['asked her if she ever had a heart attack', 'said "I\'m not sure what you\'re talking about"'])

        results = perform_intervention(intervention, self.model, effect_types=('indirect', 'direct'))
        self.assertNotAlmostEqual(results['total_effect'], 0.0)
        self.assertNotAlmostEqual(results['indirect_effect_model'], results['direct_effect_model'])

        # Now call attention_intervention_experiment where the substitutes (he, he) are identical, and verify that the
        # effects are all zero
        intervention = Intervention(
            self.tokenizer,
            'The doctor asked the nurse a question. {}',
            ['he', 'he'],
            ['asked her if she ever had a heart attack', 'said "I\'m not sure what you\'re talking about"']
        )

        results = perform_intervention(intervention, self.model, effect_types=('indirect', 'direct'))
        self.assertAlmostEqual(results['total_effect'], 0.0)
        np.testing.assert_almost_equal(results['direct_effect_head'], 0.0, decimal=5)
        np.testing.assert_almost_equal(results['indirect_effect_head'], 0.0, decimal=5)
        self.assertAlmostEqual(results['direct_effect_model'], 0.0, places=5)
        self.assertAlmostEqual(results['indirect_effect_model'], 0.0, places=5)

    def _get_probabilities_for_examples_single_token(self, context, outputs):
        """Based on previous implementation of get_probabilities_for_examples"""
        batch = torch.tensor(context).unsqueeze(0)
        logits, past = self.model.model(batch)[:2]
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        return probs[0][outputs].tolist()

    def _to_batch(self, text):
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)