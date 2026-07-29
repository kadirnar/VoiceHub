from __future__ import annotations

import unittest

import torch

from voicehub.neural import DynamicKVCache, MultiHeadAttention, RMSNorm, TransformerLayerConfig, TransformerStack


class NativeNeuralTests(unittest.TestCase):

    def test_rms_norm_accumulates_half_precision_statistics_safely(self):
        normalization = RMSNorm(4).to(dtype=torch.float16)
        inputs = torch.tensor(
            [[10_000.0, -10_000.0, 1.0, -1.0]],
            dtype=torch.float16,
        )

        output = normalization(inputs)

        self.assertEqual(output.dtype, torch.float16)
        self.assertTrue(torch.isfinite(output).all())

    def test_grouped_query_attention_cache_matches_full_causal_forward(self):
        torch.manual_seed(5)
        attention = MultiHeadAttention(
            16,
            4,
            num_key_value_heads=2,
            causal=True,
            rotary_dimension=4,
            bias=False,
        ).eval()
        hidden = torch.randn(2, 5, 16)

        full = attention(hidden).hidden_states
        cache = DynamicKVCache()
        pieces = []
        for index in range(hidden.shape[1]):
            output = attention(
                hidden[:, index:index + 1],
                cache=cache,
                layer_index=0,
                use_cache=True,
            )
            pieces.append(output.hidden_states)
        cached = torch.cat(pieces, dim=1)

        torch.testing.assert_close(cached, full, rtol=1e-5, atol=1e-5)
        self.assertEqual(cache.sequence_length(), hidden.shape[1])

    def test_boolean_padding_and_causal_masks_compose(self):
        attention = MultiHeadAttention(
            8,
            2,
            causal=True,
            bias=False,
        ).eval()
        hidden = torch.randn(1, 4, 8)
        mask = torch.tensor([[True, True, False, False]])

        output = attention(
            hidden,
            attention_mask=mask,
            output_attentions=True,
        )

        self.assertEqual(tuple(output.hidden_states.shape), (1, 4, 8))
        self.assertTrue(torch.isfinite(output.hidden_states).all())
        self.assertTrue(torch.equal(
            output.weights[..., 2:],
            torch.zeros_like(output.weights[..., 2:]),
        ))

    def test_cross_attention_cache_is_static_across_decoder_steps(self):
        config = TransformerLayerConfig(
            hidden_size=8,
            intermediate_size=16,
            num_attention_heads=2,
            activation="gelu",
            normalization="layernorm",
            causal=True,
            cross_attention=True,
        )
        stack = TransformerStack(config, num_layers=1).eval()
        encoder = torch.randn(1, 3, 8)
        cache = DynamicKVCache()
        first, cache, _ = stack(
            torch.randn(1, 1, 8),
            encoder_hidden_states=encoder,
            cache=cache,
            use_cache=True,
        )
        second, cache, _ = stack(
            torch.randn(1, 1, 8),
            encoder_hidden_states=torch.zeros_like(encoder),
            cache=cache,
            use_cache=True,
        )

        self.assertEqual(tuple(first.shape), (1, 1, 8))
        self.assertEqual(tuple(second.shape), (1, 1, 8))
        self.assertEqual(cache.sequence_length(0), 2)
        self.assertEqual(cache.sequence_length(1), 3)

    def test_cache_reorder_supports_beam_duplication(self):
        cache = DynamicKVCache()
        key = torch.arange(2 * 1 * 3 * 2).reshape(2, 1, 3, 2).float()
        cache.update(0, key, key + 1)

        cache.reorder(torch.tensor([1, 1, 0]))

        self.assertEqual(cache.get(0).key.shape[0], 3)
        torch.testing.assert_close(cache.get(0).key[0], key[1])
        torch.testing.assert_close(cache.get(0).key[1], key[1])


if __name__ == "__main__":
    unittest.main()
