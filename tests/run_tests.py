import numpy as np
import pandas as pd
from scipy import stats

from multi_drone import MultiDrone
from planner import RRTPlanner

# --- Fixed across all conditions -- do not change between environments ---
NUM_DRONES = 5
STEP_SIZE = 1.0
MAX_CONNECT_STEPS = 10
TIME_LIMIT = 20.0
N_TRIALS = 30

ENVIRONMENTS = [
    # ("easy", "environments/env_easy.yaml"),
    # ("medium", "environments/env_medium.yaml"),
    ("hard", "environments/env_hard.yaml"),
]


class HeadlessMultiDrone(MultiDrone):
    """Skip the interactive vedo window for batch runs -- see prior
    discussion, doesn't affect planning behaviour at all."""

    def _init_plot(self):
        pass

    def _update_plot(self):
        pass


def run_single_trial(env_file, seed):
    np.random.seed(seed)
    sim = HeadlessMultiDrone(num_drones=NUM_DRONES, environment_file=env_file)
    planner = RRTPlanner(
        sim, step_size=STEP_SIZE, max_connect_steps=MAX_CONNECT_STEPS,
        time_limit=TIME_LIMIT, environment_file=env_file,
    )
    planner.plan()
    return planner.last_run_stats


def run_all():
    rows = []
    for label, env_file in ENVIRONMENTS:
        for trial in range(N_TRIALS):
            result = run_single_trial(env_file, seed=trial)
            row = {"environment": label, "trial": trial}
            row.update(result)
            rows.append(row)
            print(f"[{label}] trial {trial}: success={result['success']} "
                  f"time={result['elapsed_time']:.2f}s nodes={result['nodes_expanded']}")
    return pd.DataFrame(rows)


def wilson_ci(successes, n, confidence=0.95):
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / denom
    return center - half_width, center + half_width


def mean_ci(series, confidence=0.95):
    values = series.dropna().values
    n = len(values)
    if n < 2:
        return np.nan if n == 0 else values.mean(), np.nan, np.nan
    mean = values.mean()
    sem = stats.sem(values)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - half_width, mean + half_width


def summarize(df):
    metric_cols = [
        "elapsed_time", "samples_drawn", "extend_calls",
        "motion_valid_calls", "nodes_expanded", "path_length",
    ]
    records = []
    for label, sub in df.groupby("environment", sort=False):
        n = len(sub)
        n_success = sub["success"].sum()
        succ_lo, succ_hi = wilson_ci(n_success, n)
        record = {
            "environment": label,
            "n_trials": n,
            "success_rate": n_success / n,
            "success_rate_95ci": f"[{succ_lo:.2f}, {succ_hi:.2f}]",
        }
        for col in metric_cols:
            # elapsed_time/nodes/samples/extend/motion_valid are meaningful
            # for all trials; path_length only exists for successful ones.
            source = sub[sub["success"]] if col == "path_length" else sub
            mean, lo, hi = mean_ci(source[col])
            record[f"{col}_mean"] = mean
            record[f"{col}_95ci"] = f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "n/a"
        records.append(record)
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = run_all()
    df.to_csv("results/results.csv", index=False)

    summary = summarize(df)
    summary.to_csv("results/summary.csv", index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))