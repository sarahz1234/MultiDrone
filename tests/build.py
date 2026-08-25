"""
Build three fixed environments for 5 drones.

    EASY:   5 drones, sparse scattered obstacles, no forced bottleneck,
            start/goal order unchanged -> no coordination required.

    MEDIUM: one full-height gap wide enough for ~2 drones side by side +
            moderate clutter elsewhere -> partial coordination (queueing),
            same start/goal order.

    HARD:   two full-height single-drone-width gaps in series, positioned
            off-centre from each other (forces lateral movement between them)
            + denser clutter + start/goal y-order reversed, so drones must
            actively swap relative position, not just queue in the same order
            -> genuine rearrangement
"""

import os
import yaml
import numpy as np

LO, HI = 0.0, 50.0
NUM_DRONES = 5
DRONE_DIAMETER = 0.6

def _bounds():
    return {"x": [LO, HI], "y": [LO, HI], "z": [LO, HI]}

def _full_height_gap_walls(wall_x, gap_centre_y, gap_width, thickness=1.0):

    left_size_y = (gap_centre_y - gap_width / 2.0) - LO
    right_size_y = HI - (gap_centre_y + gap_width / 2.0)