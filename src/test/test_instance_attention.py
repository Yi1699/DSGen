"""Run with: python -m unittest src.test.test_instance_attention -v"""
import subprocess
import unittest

import torch

from bldm.instance_attention import InstanceAttention
from ldm.modules.attention_dual import BasicTransformerBlock, CrossAttention, SpatialTransformer


class InstanceAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.masks = torch.zeros(1, 3, 4, 4)
        self.masks[0, 0, 0, 0] = 1
        self.masks[0, 1, 0, 1] = 1
        self.valid = torch.tensor([[1, 1, 0]])

    def masks_for(self, mode):
        return InstanceAttention(self.masks, self.valid, mode, (4,)).for_resolution(4, 4)

    def test_cross_owners_background_and_padding(self):
        cross, _ = self.masks_for("cross")
        self.assertEqual(cross[0, 0].tolist(), [True, False, False])
        self.assertEqual(cross[0, 1].tolist(), [False, True, False])
        self.assertEqual(cross[0, 2].tolist(), [True, True, False])

    def test_self_owners_and_background(self):
        _, mask = self.masks_for("self")
        self.assertTrue(mask[0, 0, 0])
        self.assertFalse(mask[0, 0, 1])
        self.assertTrue(mask[0, 0, 2])
        self.assertTrue(mask[0, 2].all())

    def test_padding_and_cross_modes_do_not_change_self(self):
        for mode in ("padding", "cross"):
            cross, mask = self.masks_for(mode)
            self.assertIsNone(mask)
            self.assertFalse(cross[:, :, 2].any())
        cross, _ = self.masks_for("padding")
        self.assertTrue(cross[:, :, :2].all())

    def test_default_self_resolutions(self):
        policy = InstanceAttention(self.masks, self.valid, "both")
        for side in (16, 32):
            self.assertIsNotNone(policy.for_resolution(side, side)[1])
        masks = policy.for_resolution(64, 64)
        self.assertIsNone(masks[1])

    def test_small_masks_and_overlap_assignment(self):
        masks = self.masks.clone()
        masks[0, 1, 0, 0] = 1
        cross, _ = InstanceAttention(masks, self.valid, "cross").for_resolution(2, 2)
        self.assertEqual(cross[0, 0].tolist(), [False, True, False])
        masks[0, 1, 0, 1] = 0
        cross, _ = InstanceAttention(masks, self.valid, "cross").for_resolution(2, 2)
        self.assertEqual(cross[0, 0].tolist(), [True, False, False])

    def test_cfg_batch_masks_stay_separate(self):
        masks = self.masks.repeat(2, 1, 1, 1)
        valid = self.valid.repeat(2, 1)
        valid[1, 1] = 0
        cross, _ = InstanceAttention(masks, valid, "cross").for_resolution(4, 4)
        self.assertEqual(cross[0, 1].tolist(), [False, True, False])
        self.assertEqual(cross[1, 1].tolist(), [True, False, False])

    def test_padding_tokens_cannot_affect_output(self):
        layer = CrossAttention(8, 12, heads=2, dim_head=4).eval()
        x, context, control = torch.randn(1, 16, 8), torch.randn(1, 5, 12), torch.randn(1, 3, 12)
        cross, _ = self.masks_for("cross")
        changed = control.clone()
        changed[:, 2] += 1000
        with torch.no_grad():
            a = layer(x, context, control, mask=cross, object_mask=cross)
            b = layer(x, context, changed, mask=cross, object_mask=cross)
        self.assertTrue(torch.equal(a, b))
        changed = control.clone()
        changed[:, 1] += 1000
        with torch.no_grad():
            b = layer(x, context, changed, mask=cross, object_mask=cross)
        self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
        self.assertFalse(torch.equal(a[:, 2], b[:, 2]))

    def test_other_instance_cannot_affect_self_query(self):
        layer = CrossAttention(8, heads=2, dim_head=4).eval()
        x = torch.randn(1, 16, 8)
        changed = x.clone()
        changed[:, 1] += 1000
        _, mask = self.masks_for("self")
        with torch.no_grad():
            a = layer(x, self_mask=mask)
            b = layer(changed, self_mask=mask)
        self.assertTrue(torch.equal(a[:, 0], b[:, 0]))
        self.assertFalse(torch.equal(a[:, 2], b[:, 2]))

    def test_empty_layout_has_zero_object_contribution(self):
        layer = CrossAttention(8, 12, heads=2, dim_head=4).eval()
        x, context, control = torch.randn(1, 16, 8), torch.randn(1, 5, 12), torch.randn(1, 3, 12)
        for mode in ("padding", "cross", "self", "both"):
            cross, mask = InstanceAttention(self.masks, torch.zeros_like(self.valid), mode, (4,)).for_resolution(4, 4)
            with torch.no_grad():
                a = layer(x, context, control, mask=cross, object_mask=cross)
                b = layer(x, context)
            self.assertTrue(torch.isfinite(a).all())
            self.assertTrue(torch.equal(a, b))
            if mask is not None:
                self.assertTrue(mask.all())

    def test_spatial_transformer_routes_masks_at_feature_resolution(self):
        layer = SpatialTransformer(32, 2, 8, context_dim=48).eval()
        torch.nn.init.normal_(layer.proj_out.weight, std=0.1)
        x, context, control = torch.randn(1, 32, 16, 16), torch.randn(1, 5, 48), torch.randn(1, 3, 48)
        with torch.no_grad():
            baseline = layer(x, context, control, self.masks.amax(1))
            for mode in ("padding", "cross", "self", "both"):
                policy = InstanceAttention(self.masks, self.valid, mode)
                result = layer(x, context, control, policy)
                self.assertEqual(result.shape, x.shape)
                self.assertTrue(torch.isfinite(result).all())
                self.assertFalse(torch.equal(baseline, result))

    def test_baseline_matches_main_bit_for_bit(self):
        source = subprocess.check_output(["git", "show", "origin/main:ldm/modules/attention_dual.py"], text=True)
        original = {}
        exec(compile(source, "original_attention_dual.py", "exec"), original)
        old = original["BasicTransformerBlock"](32, 2, 8, context_dim=48, checkpoint=False).eval()
        new = BasicTransformerBlock(32, 2, 8, context_dim=48, checkpoint=False).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        x, context, control = torch.randn(2, 16, 32), torch.randn(2, 5, 48), torch.randn(2, 3, 48)
        mask = torch.ones(2, 4, 4)
        with torch.no_grad():
            a, b = old(x, context, control, mask), new(x, context, control, mask)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))


if __name__ == "__main__":
    unittest.main()
