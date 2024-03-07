import math
import numpy as np

from timeit import default_timer as timer


class ACO:
    def __init__(self, nodes, distances, ants_to_nodes_ratio=0.5, alpha=5, beta=3, q=1000, evaporation=0.1, pheromones_min=1):
        self.nodes = nodes
        self.ants_to_nodes_ratio = ants_to_nodes_ratio
        self.alpha = alpha
        self.beta = beta
        self.q = q
        self.evaporation = evaporation
        self.pheromones_min = pheromones_min

        self.size = len(nodes)

        self.ants_count = math.floor(self.size * self.ants_to_nodes_ratio)
        self.distances = distances
        self.visibility = 1 / (self.distances ** 3)
        self.visibility[self.visibility == np.inf] = 0

        self.pheromones = np.ones((self.size, self.size))

        self.best_score = float("-inf")
        self.best_path = None

        self.history = []

    def get_ant_scores(self, paths):
        each_ant_perfect_weights_sum = np.count_nonzero(
            self.distances[paths[:, 1:], paths[:, :-1]] == 1, axis=1)

        each_ant_full_distance = np.sum(
            self.distances[paths[:, 1:], paths[:, :-1]], axis=1) - each_ant_perfect_weights_sum - self.size

        each_ant_score = 1000 * each_ant_perfect_weights_sum - \
            0 * each_ant_full_distance

        return each_ant_score

    def step(self):
        size = self.size
        unvisited = np.ones((self.ants_count, size))
        paths = np.zeros((self.ants_count, size), int) - 1

        current_nodes = np.zeros(self.ants_count, dtype=int)

        unvisited[np.arange(self.ants_count), current_nodes] = 0
        paths[:, 0] = current_nodes

        probabilities = (self.pheromones ** self.alpha) * \
            (self.visibility ** self.beta)

        for i in range(size - 1):
            next_probabilities = unvisited * probabilities[current_nodes, :]
            next_probabilities *= (1 /
                                   np.sum(next_probabilities, axis=1)[:, None])

            next_nodes = [np.random.choice(np.arange(size), p=next_probabilities[a])
                          for a in range(self.ants_count)]

            unvisited[np.arange(self.ants_count), next_nodes] = 0
            paths[:, i+1] = next_nodes

            current_nodes = next_nodes

        ant_scores = self.get_ant_scores(paths)

        best_ant_path = np.argmax(ant_scores)
        if self.best_score < ant_scores[best_ant_path]:
            self.best_score = ant_scores[best_ant_path]
            self.best_path = paths[best_ant_path]

        self.pheromones = (1 - self.evaporation) * self.pheromones
        self.pheromones[self.pheromones <
                        self.pheromones_min] = self.pheromones_min

        np.add.at(self.pheromones,
                  (paths[:, 1:], paths[:, :-1]), self.q / self.distances[paths[:, 1:], paths[:, :-1]])
        np.add.at(self.pheromones,
                  (paths[:, :-1], paths[:, 1:]), self.q / self.distances[paths[:, :-1], paths[:, 1:]])

        self.history.append(ant_scores[best_ant_path])

    def run(self, dropout_threshold=3, timeout=300, max_iterations=None, after_step=None):
        time_start = timer()
        no_improvement_count = 0
        best_score = float("-inf")

        while True:
            self.step()
            if after_step:
                after_step(len(self.history))

            if max_iterations:
                if len(self.history) == max_iterations:
                    print(f"Terminating: MAX ITERATIONS ({max_iterations})")

            if timeout:
                time_elapsed = timer() - time_start
                if time_elapsed > timeout:
                    print(f"Terminating: TIMEOUT ({timeout}s)")
                    return

            if self.best_score > best_score:
                best_score = self.best_score
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count == dropout_threshold:
                    print(
                        f"Terminating: NO IMPROVEMENT for {dropout_threshold} iterations in a row")
                    return
