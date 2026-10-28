"""Inference-only instance masks for AeroGen's existing attention layers."""

import torch
import torch.nn.functional as F


class InstanceAttention:
    def __init__(self, masks, valid, mode, self_resolutions=(16, 32)):
        self.masks = masks.float() * valid[:, :, None, None]
        self.valid = valid.bool()
        self.mode = mode
        self.self_resolutions = self_resolutions
        self.cache = {}

    def for_resolution(self, height, width):
        key = (height, width)
        if key in self.cache:
            return self.cache[key]
        coverage = F.interpolate(self.masks, size=key, mode="area").flatten(2)
        amount, owner = coverage.max(dim=1)
        owner = owner.masked_fill(amount == 0, -1)
        background = owner == -1
        object_mask = self.valid[:, None, :].expand(-1, height * width, -1)
        if self.mode in ("cross", "both"):
            indices = torch.arange(self.valid.shape[1], device=owner.device)
            object_mask = object_mask & (background[:, :, None] | (owner[:, :, None] == indices))
        self_mask = None
        if self.mode in ("self", "both") and height in self.self_resolutions and width in self.self_resolutions:
            self_mask = (background[:, :, None] | background[:, None, :]
                         | (owner[:, :, None] == owner[:, None, :]))
        self.cache[key] = object_mask, self_mask
        return object_mask, self_mask
