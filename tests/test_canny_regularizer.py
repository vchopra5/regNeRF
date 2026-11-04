"""Check the Canny-based depth rule without training a model."""
import unittest
from unittest.mock import patch

import torch

from canny_regularizer import HALO, canny_depth_weights, canny_edge_l1_loss
from train import PATCH_SIZE, depth_patch_loss, virtual_patch_rgb


class CannyDepthTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.side = PATCH_SIZE + 2 * HALO

    def guide(self):
        return torch.zeros(2, self.side, self.side, 3, device=self.device)

    def test_flat_guide_keeps_uniform_loss(self):
        rgb = self.guide() + 0.4
        weights = canny_depth_weights(rgb)
        depth = torch.randn(2, PATCH_SIZE, PATCH_SIZE, device=self.device)
        opacity = torch.ones_like(depth)
        torch.testing.assert_close(depth_patch_loss(depth, opacity, weights),
                                   depth_patch_loss(depth, opacity))

    def test_edge_direction_and_floor(self):
        rgb = self.guide()
        rgb[:, :, self.side // 2:] = 1
        wx, wy = canny_depth_weights(rgb)
        self.assertLess(wx.min().item(), 0.5)
        self.assertGreaterEqual(wx.min().item(), 0.05)
        torch.testing.assert_close(wy, torch.ones_like(wy))
        tx, ty = canny_depth_weights(rgb.transpose(1, 2))
        torch.testing.assert_close(tx, wy.transpose(1, 2))
        torch.testing.assert_close(ty, wx.transpose(1, 2))

    def test_loss_has_depth_gradients_only(self):
        rgb = self.guide().requires_grad_()
        depth = torch.randn(2, PATCH_SIZE, PATCH_SIZE,
                            device=self.device, requires_grad=True)
        opacity = torch.ones_like(depth, requires_grad=True)
        loss = depth_patch_loss(depth, opacity, canny_depth_weights(rgb))
        loss.backward()
        self.assertIsNone(rgb.grad)
        self.assertIsNone(opacity.grad)
        self.assertTrue(torch.isfinite(depth.grad).all())
        self.assertGreater(depth.grad.abs().sum().item(), 0)

    def test_constant_depth_has_zero_loss(self):
        depth = torch.ones(2, PATCH_SIZE, PATCH_SIZE, device=self.device)
        self.assertEqual(depth_patch_loss(depth, depth,
                         canny_depth_weights(self.guide())).item(), 0)

    def test_l1_penalizes_missing_edge_and_starts_separation(self):
        depth = torch.ones(1, 2, 2, device=self.device, requires_grad=True)
        wx = torch.full((1, 1, 1), 0.05, device=self.device)
        wy = torch.ones_like(wx)
        loss = canny_edge_l1_loss(depth, torch.ones_like(depth), (wx, wy))
        self.assertAlmostEqual(loss.item(), 0.1, places=6)
        loss.backward()
        changed = depth.detach() - 0.01 * depth.grad
        self.assertGreater((changed[0, 0, 0] - changed[0, 0, 1]).abs().item(), 0)
        self.assertLess(canny_edge_l1_loss(changed, torch.ones_like(depth), (wx, wy)).item(), loss.item())

    def test_l1_is_zero_without_edges_or_with_sufficient_depth_change(self):
        weights = torch.ones(1, 1, 1, device=self.device)
        depth = torch.ones(1, 2, 2, device=self.device)
        self.assertEqual(canny_edge_l1_loss(depth, depth, (weights, weights)).item(), 0)
        for sign in (-1, 1):
            depth = torch.tensor([[[0., sign * 0.2], [0., 0.]]], device=self.device)
            self.assertEqual(canny_edge_l1_loss(depth, torch.ones_like(depth),
                             (weights * 0.05, weights)).item(), 0)

    def test_render_restores_mode_and_uses_halo(self):
        model = torch.nn.Linear(1, 1).to(self.device)
        poses = torch.zeros(2, 3, 4, device=self.device)
        origin = torch.zeros(2, 1, device=self.device, dtype=torch.long)
        fields = [torch.zeros(2, self.side**2, 3, device=self.device)] * 3

        def render(*args):
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
            return torch.ones(2 * self.side**2, 3, device=self.device)

        with patch("train.pixel_rays", return_value=fields) as rays:
            with patch("train.render_rays", side_effect=render):
                result = virtual_patch_rgb(model, 32, 32, None, poses,
                                           origin, origin, HALO)
            self.assertEqual(result.shape, (2, self.side, self.side, 3))
            self.assertTrue(model.training)
            self.assertGreaterEqual(rays.call_args.args[-2].min().item(), 0)
            with patch("train.render_rays", side_effect=RuntimeError("test")):
                with self.assertRaises(RuntimeError):
                    virtual_patch_rgb(model, 32, 32, None, poses,
                                      origin, origin, HALO)
            self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
