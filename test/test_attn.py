import unittest

from gpt2_attention import AttentionOverride
from experiment import Model
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class AttnTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def test_attention_override(self):

        # TEST 1: Overlay attention from same input and verify that output doesn't change

        text = 'The teacher said that he'
        x = self._to_batch(text)
        layer = 2
        seq_len = x.shape[1]
        outputs = self.model.model(x)
        lm_logits = outputs[0] # Shape: batch_size, seq_len, vocab_size

        attn = outputs[-1][layer] # Shape: batch_size, num_heads, seq_len, seq_len
        self.assertEqual(attn.shape[0], 1)
        self.assertEqual(attn.shape[1], self.model.num_heads)
        self.assertEqual(attn.shape[2], seq_len)
        self.assertEqual(attn.shape[3], seq_len)

        attn_override = attn.clone()
        attn_override_mask = torch.ones_like(attn_override, dtype=torch.uint8)

        def intervention_hook(module, input, outputs):
            attention_override_module = AttentionOverride(module, attn_override, attn_override_mask)
            outputs[:] = attention_override_module(*input)

        hook = self.model.model.transformer.h[layer].attn.register_forward_hook(intervention_hook)
        with torch.no_grad():
            lm_logits2 = self.model.model(x)[0]
        hook.remove()
        self.assertTrue(torch.allclose(lm_logits, lm_logits2))

        # TEST 2: Override attention with zeros for one of the heads and verify that output does change

        head = 5
        attn_override = torch.zeros(1, self.model.num_heads, seq_len, seq_len)
        attn_override_mask = torch.zeros_like(attn_override, dtype=torch.uint8)
        attn_override_mask[0][head] = 1

        hook = self.model.model.transformer.h[layer].attn.register_forward_hook(intervention_hook)
        with torch.no_grad():
            lm_logits3 = self.model.model(x)[0]
        hook.remove()
        self.assertFalse(torch.allclose(lm_logits, lm_logits3))

        # TEST 3: Override attention with zeros for one of the heads, and verify that gives same output as prune_heads

        prune_model = GPT2LMHeadModel.from_pretrained('gpt2')
        prune_model.prune_heads({layer: [head]})
        lm_logits4 = prune_model(x)[0]
        self.assertTrue(torch.allclose(lm_logits3, lm_logits4))

        # TEST 4: Override attention for subsequence

        attn_text = 'The doctor said that she' # Text from which attention override will be generated
        outputs = self.model.model(self._to_batch(attn_text))
        layer = 2
        attn_override = outputs[-1][layer]
        attn_override_mask = torch.ones_like(attn_override, dtype=torch.uint8)

        def intervention_hook2(module, input, outputs):
            attention_override_module = AttentionOverride(module, attn_override, attn_override_mask)
            outputs[:] = attention_override_module(*input)

        # Override attention from subsequence of same sequence and verify that output is the same
        text = 'The doctor said that she went to the store'
        outputs = self.model.model(self._to_batch(text))
        lm_logits2 = outputs[0]
        hook = self.model.model.transformer.h[layer].attn.register_forward_hook(intervention_hook2)
        with torch.no_grad():
            outputs = self.model.model(self._to_batch(text))
            lm_logits3 = outputs[0]
        hook.remove()
        self.assertTrue(torch.allclose(lm_logits2, lm_logits3))

        # Override attention from different subsequence and verify that output changes
        for text2 in ['The doctor said that he went to the store', 'That doctor said that she went to the store']:
            hook = self.model.model.transformer.h[layer].attn.register_forward_hook(intervention_hook2)
            with torch.no_grad():
                outputs = self.model.model(self._to_batch(text2))
                lm_logits4 = outputs[0]
            hook.remove()
            self.assertFalse(torch.allclose(lm_logits2, lm_logits4))



    def _to_batch(self, text):
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)