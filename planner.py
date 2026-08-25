import time
import numpy as np

class Tree:
    def __init__(self, root, root_type):
        self.nodes = [np.array(root, dtype=np.float32)]
        self.parents = [-1]
        self.root_type = root_type # "start" or "goal"

    def add_node(self, q, parent_index):
        self.nodes.append(np.array(q, dtype=np.float32))
        self.parents.append(parent_index)
        return len(self.nodes) - 1

    def get_path_to_root(self, node_index: int):
        path = []
        while node_index != -1:
            path.append(self.nodes[node_index])
            node_index = self.parents[node_index]
        path.reverse()
        return path

class RRTPlanner:
    def __init__(
            self,
            simulator,
            step_size,
            max_connect_steps,
            time_limit=20.0,
            environment_file=None
    ):
        self.sim = simulator
        self.step_size = step_size
        self.max_connect_steps = max_connect_steps
        self.time_limit = time_limit
        self.bounds = self._resolve_bounds(simulator, environment_file)
        self.num_drones = simulator.N


    @staticmethod
    def _resolve_bounds(simulator, environment_file):
        if hasattr(simulator, "_bounds"):
            return simulator._bounds

        if environment_file is not None:
            from multi_drone import load_obstacles_from_yaml
            _, _, _, bounds = load_obstacles_from_yaml(environment_file, num_drones=simulator.N)

        raise RuntimeError("Could not determine environment bounds: simulator has no"
                           "_bounds attribute, and no envrionment_file was provided to"
                           "fall back on. Pass envrionment_file=... to RRTPlanner.")


    def distance(self, q1, q2):
        return np.linalg.norm(q1.flatten() - q2.flatten())

    def sample_config(self):
        while True:
            q = np.random.uniform(
                low=self.bounds[:, 0],
                high=self.bounds[:, 1],
                size=(self.num_drones, 3)
            ).astype(np.float32)
            self.samples_drawn += 1 # count every draw, including rejected ones
            if self.sim.is_valid(q):
                return q

    def nearest(self, tree, q_target) -> int:
        """
        Linear scan for the nearest node in the tree to q_target.
        """
        dists = [self.distance(q, q_target) for q in tree.nodes]
        return int(np.argmin(dists))

    def steer(self, q_from, q_target):
        """
        Move at most self.step_size steps from q_from to q_target along
        the straight line in the flattened joint space. If q_target is
        already within step_size, return it directly.
        """
        delta = q_target.flatten() - q_from.flatten()
        dist = np.linalg.norm(delta)
        if dist <= self.step_size:
            return q_target.copy()
        q_new_flat = q_from.flatten() + self.step_size * (delta / dist)
        return q_new_flat.reshape(q_from.shape)

    def extend(self, tree, q_target):
        """
        Returns (status, node_index):
            TRAPPED -- no collision-free step could be taken, node_index=None
            ADVANCED -- moved step_size toward q_target
            REACHED -- connect exactly to q_target
        """
        self.extend_calls += 1
        near_idx = self.nearest(tree, q_target)
        q_near = tree.nodes[near_idx]
        q_new = self.steer(q_near, q_target)

        self.motion_valid_calls += 1
        if not self.sim.motion_valid(q_near, q_new):
            return "TRAPPED", None

        new_idx = tree.add_node(q_new, near_idx)

        if np.allclose(q_new, q_target):
            return "REACHED", new_idx

        return "ADVANCED", new_idx

    def connect(self, tree, q_target):
        """
        Repeatedly extend 'tree' toward q_target, up to max_connect_steps
        times. Stops early on TRAPPED or REACHED.
        """
        node_index = None
        for _ in range(self.max_connect_steps):
            status, node_index = self.extend(tree, q_target)
            if status != "ADVANCED":
                return status, node_index
        return "ADVANCED", node_index

    def combine_paths(self, tree_a, node_a, tree_b, node_b):
        """
        Combine the two half-paths together at the point where the trees met.
        """
        path_a = tree_a.get_path_to_root(node_a) # root(tree_a) -> meeting point
        path_b = tree_b.get_path_to_root(node_b) # root(tree_b) -> meeting point

        if tree_a.root_type == "start":
            full_path = path_a + path_b[::-1][1:]
        else:
            full_path = path_b + path_a[::-1][1:]

        return full_path

    def plan(self):
        # Instrumentation counters, reset per plan() call
        self.samples_drawn = 0
        self.extend_calls = 0
        self.motion_valid_calls = 0

        q_start = self.sim.initial_configuration
        q_goal = self.sim.goal_positions

        start_tree = Tree(q_start, "start")
        goal_tree = Tree(q_goal, "goal")

        tree_a, tree_b = start_tree, goal_tree

        start_time = time.time()
        result = None
        while time.time() - start_time < self.time_limit:
            q_rand = self.sample_config()
            status_a, node_a = self.extend(tree_a, q_rand)

            if status_a != "TRAPPED":
                q_new = tree_a.nodes[node_a]
                status_b, node_b = self.connect(tree_b, q_new)

                if status_b == "REACHED":
                    result = self.combine_paths(tree_a, node_a, tree_b, node_b)
                    break

            tree_a, tree_b = tree_b, tree_a

        # Record experiment statistics
        self.last_run_stats = {
            "success": result is not None,
            "elapsed_time": time.time() - start_time,
            "samples_drawn": self.samples_drawn,
            "extend_calls": self.extend_calls,
            "motion_valid_calls": self.motion_valid_calls,
            "nodes_expanded": len(start_tree.nodes) + len(goal_tree.nodes),
            "path_length": (
                sum(self.distance(result[i], result[i + 1]) for i in range(len(result) - 1))
                if result is not None else None
            )
        }

        return result

if __name__ == "__main__":
    from multi_drone import MultiDrone

    env_file = "environment.yaml"
    sim = MultiDrone(num_drones=2, environment_file=env_file)
    planner = RRTPlanner(
        sim,
        step_size=1.0,
        max_connect_steps=10,
        time_limit=20.0,
        environment_file=env_file,
    )

    path = planner.plan()
    if path is not None:
        sim.visualize_paths(path)
    else:
        print("failure")