# -*- coding: utf-8 -*-
"""Compatibility shim: Zhizi inversion bridge → Physics Decoder.

Prefer ``from hnf.physics_decoder import PhysicsDecoder``.
"""

from __future__ import annotations

from hnf.physics_decoder import (  # noqa: F401
    PhysicsDecoder,
    ZhiziInversionBridge,
    features_to_head_inputs,
    load_inversion_bridge_from_checkpoint,
    load_physics_decoder_from_checkpoint,
    load_physics_head_state,
    pick_times_from_logits,
)

__all__ = [
    "PhysicsDecoder",
    "ZhiziInversionBridge",
    "features_to_head_inputs",
    "load_inversion_bridge_from_checkpoint",
    "load_physics_decoder_from_checkpoint",
    "load_physics_head_state",
    "pick_times_from_logits",
]
