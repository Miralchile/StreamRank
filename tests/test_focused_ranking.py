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
        return {
            "user_ids": torch.tensor([2, 3]),
            "item_ids": torch.tensor([2, 4]),
            "tab_ids": torch.tensor([1, 2]),
            "histories": torch.tensor([[0, 2, 3], [0, 0, 4]]),
            "history_mask": torch.tensor([[False, True, True], [False, False, True]]),
            "numeric": torch.zeros((2, 8)),
            "is_click": torch.tensor([1.0, 0.0]),
            "long_view": torch.tensor([1.0, 0.0]),
            "is_like": torch.tensor([0.0, 0.0]),
            "is_hate": torch.tensor([0.0, 1.0]),
        }

    def test_all_focused_variants_have_task_outputs(self):
        batch = self._batch()
        for architecture, task_layer in (
            ("deepfm", "shared_bottom"),
            ("din", "shared_bottom"),
            ("din", "mmoe"),
        ):
            model = FocusedRanker(architecture, task_layer, 10, 10, embedding_dim=8, hidden_dim=12)
            logits = model(batch)
            self.assertEqual(set(logits), {"is_click", "long_view", "is_like", "is_hate"})
            self.assertEqual(tuple(logits["is_click"].shape), (2,))
            loss = weighted_multitask_loss(logits, batch, {task: 1.0 for task in logits})
            self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
