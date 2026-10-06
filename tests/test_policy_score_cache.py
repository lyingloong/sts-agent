"""Independent forward pass regression for shared-cache candidate scoring."""

from __future__ import annotations

import unittest


class CacheScoringTests(unittest.TestCase):
    def test_qwen_scores_match_full_forward_and_do_not_depend_on_iteration_order(self):
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
        except ImportError:
            self.skipTest("optional torch/transformers are not installed")
        from sts_agent.policy.local_qwen import QwenRanker

        class Tokenizer:
            def __call__(self, text, add_special_tokens=False, return_tensors=None):
                tokens = [1, 2, 3]
                if text.endswith("A1\n"):
                    tokens += [4, 5, 6]
                elif text.endswith("A2\n"):
                    tokens += [4, 7, 6]
                return {"input_ids": torch.tensor([tokens]) if return_tensors else tokens}

        torch.manual_seed(17)
        config = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                             num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        model = Qwen3ForCausalLM(config).eval()
        ranker = QwenRanker.__new__(QwenRanker)
        ranker.model, ranker.torch, ranker.tokenizer, ranker.model_device = model, torch, Tokenizer(), torch.device("cpu")
        ranker._chat_prefix = lambda prompt: "prefix"
        actions = [{"id": "A1", "label": "One"}, {"id": "A2", "label": "Two"}]
        forward = {x["action_id"]: x["logprob"] for x in ranker.score({}, actions)}
        reverse = {x["action_id"]: x["logprob"] for x in ranker.score({}, list(reversed(actions)))}
        with torch.inference_mode():
            for alias, suffix in (("A1", [4, 5, 6]), ("A2", [4, 7, 6])):
                logits = model(input_ids=torch.tensor([[1, 2, 3] + suffix]), use_cache=False).logits[0]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                expected = sum(logprobs[2 + i, token].item() for i, token in enumerate(suffix))
                self.assertAlmostEqual(forward[alias], expected, places=5)
                self.assertAlmostEqual(reverse[alias], expected, places=5)


if __name__ == "__main__":
    unittest.main()
