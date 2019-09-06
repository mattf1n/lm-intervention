import unittest
from experiment import Model
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel


class AttnTest(unittest.TestCase):

    def setUp(self):
        self.model = Model(output_attentions=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def test_recompute_attention_outputs(self):

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

        def update_output(module, input, outputs):
            outputs[0] = self.model.get_attention_output(input[0], module, attn_override, attn_override_mask)

        hook = self.model.model.transformer.h[layer].attn.register_forward_hook(update_output)
        with torch.no_grad():
            lm_logits2 = self.model.model(x)[0]
        hook.remove()
        self.assertTrue(torch.allclose(lm_logits, lm_logits2))

        # TEST 2: Override attention with zeros for one of the heads and verify that output does change

        head = 5
        attn_override = torch.zeros(1, self.model.num_heads, seq_len, seq_len)
        attn_override_mask = torch.zeros_like(attn_override, dtype=torch.uint8)
        attn_override_mask[0][head] = torch.ones((seq_len, seq_len), dtype=torch.uint8)

        hook = self.model.model.transformer.h[layer].attn.register_forward_hook(update_output)
        with torch.no_grad():
            lm_logits3 = self.model.model(x)[0]
        hook.remove()
        self.assertFalse(torch.allclose(lm_logits, lm_logits3))

        # TEST 3: Override attention with zeros for one of the heads, and verify that gives same output as prune_heads

        prune_model = GPT2LMHeadModel.from_pretrained('gpt2')
        prune_model.prune_heads({layer: [head]})
        lm_logits4 = prune_model(x)[0]
        self.assertTrue(torch.allclose(lm_logits3, lm_logits4))

    def _to_batch(self, text):
        encoded = self.tokenizer.encode(text)
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)