#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deprecated alias — use sweep_fluid_baseline3d4d.py (U-Net is baseline, not SOTA)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_fluid_baseline3d4d import main

if __name__ == "__main__":
    main()
