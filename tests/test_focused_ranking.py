from __future__ import annotations

import unittest

try:
    import torch

    from streamrank.focused.models import FocusedRanker, weighted_multitask_loss

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "ml extra is not installed")
class FocusedRankingTest(unittest.TestCase):
    def _batch(self):
        # The third row has an entirely empty history: encoders must stay finite
        # (fully masked attention rows are the classic NaN pitfall).
        return {
            "user_ids": torch.tensor([2, 3, 1]),
            "item_ids": torch.tensor([2, 4, 5]),
            "tab_ids": torch.tensor([1, 2, 0]),
            "histories": torch.tensor([[0, 2, 3], [0, 0, 4], [0, 0, 0]]),
            "history_mask": torch.tensor(
                [[False, True, True], [False, False, True], [False, False, False]]
            ),
            "numeric": torch.zeros((3, 8)),
            "is_click": torch.tensor([1.0, 0.0, 1.0]),
            "long_view": torch.tensor([1.0, 0.0, 0.0]),
            "is_like": torch.tensor([0.0, 0.0, 1.0]),
            "is_hate": torch.tensor([0.0, 1.0, 0.0]),
        }

    def test_all_focused_variants_have_finite_task_outputs(self):
        batch = self._batch()
        for architecture, task_layer in (
            ("deepfm", "shared_bottom"),
            ("din", "shared_bottom"),
            ("din", "mmoe"),
            ("sasrec", "shared_bottom"),
            ("bst", "shared_bottom"),
            ("autoint", "mmoe"),
        ):
            model = FocusedRanker(
                architecture,
                task_layer,
                10,
                10,
                embedding_dim=8,
                hidden_dim=12,
                num_heads=2,
                num_layers=1,
            )
            model.eval()
            logits = model(batch)
            self.assertEqual(set(logits), {"is_click", "long_view", "is_like", "is_hate"})
            self.assertEqual(tuple(logits["is_click"].shape), (3,))
            for task, output in logits.items():
                self.assertTrue(
                    torch.isfinite(output).all(), f"{architecture}/{task} produced non-finite"
                )
            loss = weighted_multitask_loss(logits, batch, {task: 1.0 for task in logits})
            self.assertTrue(torch.isfinite(loss))

    def test_rejects_unknown_architecture(self):
        with self.assertRaises(ValueError):
            FocusedRanker("transformer-xl", "shared_bottom", 10, 10)


if __name__ == "__main__":
    unittest.main()
