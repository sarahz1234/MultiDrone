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

def _full_height_gap_walls(wall_x, gap_center_y, gap_width, thickness=1.0):
    """
    Two box obstacles spanning the FULL z-range, split along y with a gap
    of gap_width centered at gap_center_y. This is what actually forces
    horizontal squeezing instead of over-flight.
    """
    left_size_y = (gap_center_y - gap_width / 2.0) - LO
    right_size_y = HI - (gap_center_y + gap_width / 2.0)
    assert left_size_y > 0 and right_size_y > 0, "gap too wide / off-center for these bounds"

    z_mid = (LO + HI) / 2.0
    return [
        {
            "type": "box",
            "position": [wall_x, LO + left_size_y / 2.0, z_mid],
            "size": [thickness, left_size_y, HI - LO],
            "color": "red",
        },
        {
            "type": "box",
            "position": [wall_x, gap_center_y + gap_width / 2.0 + right_size_y / 2.0, z_mid],
            "size": [thickness, right_size_y, HI - LO],
            "color": "red",
        },
    ]

def _clutter(rng, count, x_range, size=2.0):
    """
    Randomly placed short obstacles.
    """
    obstacles = []
    while len(obstacles) < count:
        pos = rng.uniform([x_range[0], LO + 3, LO + 3], [x_range[1], HI - 3, HI / 2])
        kind = rng.choice(["box", "sphere", "cylinder"])

        if kind == "box":
            obstacles.append({
                "type": "box",
                "position": pos.tolist(),
                "size": [size, size, size],
                "color": "gray",
            })
        elif kind == "sphere":
            radius = float(rng.uniform(0.8, 1.5))
            obstacles.append({
                "type": "sphere",
                "position": pos.tolist(),
                "radius": radius,
                "color": "gray",
            })
        else:  # cylinder -- vertical axis, endpoints straddle pos in z
            radius = float(rng.uniform(0.6, 1.2))
            half_height = float(rng.uniform(1.0, 2.5))
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            p1 = [px, py, pz - half_height]
            p2 = [px, py, pz + half_height]
            obstacles.append({
                "type": "cylinder",
                "endpoints": [p1, p2],
                "radius": radius,
                "color": "gray",
            })
    return obstacles


def _fmt_list(values):
    """Flow-style inline list matching the reference format: [1, 2, 3]."""
    return "[" + ", ".join(_fmt_num(v) for v in values) + "]"


def _fmt_num(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:.4g}"


def _write(path, obstacles, start_ys, goal_ys, goal_x, goal_radius=1.0):
    """
    Hand-formats the YAML text to match the reference environment.yaml
    style exactly: flow-style inline lists ([x, y, z], not block '- x'),
    inline comments, rotation present for box/cylinder but omitted for
    sphere (matching the reference's sphere entry). yaml.safe_dump's
    default block style doesn't produce this, so we write the text
    directly rather than relying on the dumper's formatting.
    """
    z_mid = (LO + HI) / 2.0
    initial_configuration = [[LO + 2.0, float(y), z_mid] for y in start_ys]
    goals = [[goal_x, float(y), z_mid] for y in goal_ys]

    lines = []
    lines.append("bounds:")
    lines.append(f"  x: {_fmt_list([LO, HI])}")
    lines.append(f"  y: {_fmt_list([LO, HI])}")
    lines.append(f"  z: {_fmt_list([LO, HI])}")
    lines.append("")

    ic_str = "[" + ", ".join(_fmt_list(p) for p in initial_configuration) + "]"
    lines.append(f"initial_configuration: {ic_str} ")
    lines.append("")

    lines.append("obstacles:")
    for obj in obstacles:
        if obj["type"] == "box":
            lines.append(f"  - type: box")
            lines.append(f"    position: {_fmt_list(obj['position'])} ")
            lines.append(f"    size: {_fmt_list(obj['size'])} ")
            lines.append(f"    rotation: {_fmt_list(obj.get('rotation', [0, 0, 0]))} ")
            lines.append(f"    color: {obj.get('color', 'gray')}")
        elif obj["type"] == "cylinder":
            p1, p2 = obj["endpoints"]
            lines.append(f"  - type: cylinder")
            lines.append(f"    endpoints: [{_fmt_list(p1)}, {_fmt_list(p2)}] ")
            lines.append(f"    radius: {_fmt_num(obj['radius'])} ")
            lines.append(f"    rotation: {_fmt_list(obj.get('rotation', [0, 0, 0]))}")
            lines.append(f"    color: {obj.get('color', 'gray')}")
        elif obj["type"] == "sphere":
            lines.append(f"  - type: sphere")
            lines.append(f"    position: {_fmt_list(obj['position'])}")
            lines.append(f"    radius: {_fmt_num(obj['radius'])} ")
            lines.append(f"    color: {obj.get('color', 'gray')}")
        lines.append("")

    lines.append("goals:")
    for i, g in enumerate(goals):
        lines.append(f"  - position: {_fmt_list(g)} # Goal {i + 1}")
        lines.append(f"    radius: {_fmt_num(goal_radius)}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_easy(path):
    rng = np.random.default_rng(46)
    start_ys = np.linspace(LO + 5, HI - 5, NUM_DRONES)
    goal_ys = start_ys.copy()  # same order -- no coordination required
    obstacles = _clutter(rng, count=4, x_range=(LO + 10, HI - 10))
    _write(path, obstacles, start_ys, goal_ys, goal_x=HI - 2.0)


def build_medium(path):
    rng = np.random.default_rng(35)
    start_ys = np.linspace(LO + 5, HI - 5, NUM_DRONES)
    goal_ys = start_ys.copy()  # same order -- passage forces queueing, not swapping
    # Gap wide enough for ~2 drones at a time:
    #   2*diameter + inter-drone clearance + 2*wall clearance
    #   ~= 2*0.6 + 0.6 + 2*0.3 = 2.4, so use 3.0 for a comfortable "fits 2" gap.
    gap_width = 3.0
    obstacles = _full_height_gap_walls(wall_x=25.0, gap_center_y=25.0, gap_width=gap_width)
    obstacles += _clutter(rng, count=8, x_range=(LO + 3, HI - 3))
    _write(path, obstacles, start_ys, goal_ys, goal_x=HI - 2.0)


def build_hard(path, gap_width=2.0, num_bottlenecks=1, clutter_count=20, full_reversal=False):
    rng = np.random.default_rng(92)
    start_ys = np.linspace(LO + 5, HI - 5, NUM_DRONES)

    if full_reversal:
        goal_ys = start_ys[::-1].copy()
    else:
        # Swap only the two outermost drones; middle three keep their
        # relative order. Still forces active coordination (drone 1 and
        # drone 5 must pass each other somewhere), just not a full invert.
        goal_ys = start_ys.copy()
        goal_ys[0], goal_ys[-1] = start_ys[-1], start_ys[0]

    obstacles = []
    if num_bottlenecks >= 1:
        obstacles += _full_height_gap_walls(wall_x=20.0, gap_center_y=25.0, gap_width=gap_width)
    if num_bottlenecks >= 2:
        obstacles += _full_height_gap_walls(wall_x=32.0, gap_center_y=30.0, gap_width=gap_width)

    obstacles += _clutter(rng, count=clutter_count, x_range=(LO + 3, HI - 3))
    _write(path, obstacles, start_ys, goal_ys, goal_x=HI - 2.0)


if __name__ == "__main__":
    os.makedirs("environments", exist_ok=True)
    build_easy("environments/env_easy.yaml")
    build_medium("environments/env_medium.yaml")
    build_hard("environments/env_hard.yaml")
    print("Wrote environments/env_easy.yaml, env_medium.yaml, env_hard.yaml")


