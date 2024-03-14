import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
from bispy import compute_maximum_bisimulation
from tqdm import tqdm
from time import perf_counter
from typing import Dict, Tuple, Any, List
import pickle
from itertools import product
import concurrent.futures
from collections import defaultdict
import time


def generate_binary_sparse_matrix(rows, cols, density=0.1):
    """
    Generate a binary sparse matrix.

    Parameters:
    rows (int): Number of rows in the matrix.
    cols (int): Number of columns in the matrix.
    density (float): Fraction of elements that are non-zero. Should be between 0 and 1.

    Returns:
    dict: A dictionary representing the binary sparse matrix.
    """
    if not (0 <= density <= 1):
        raise ValueError("Density must be between 0 and 1")
    rng = np.random.default_rng()

    num_nonzero = int(rows * cols * density)

    row = rng.integers(0, rows - 1, num_nonzero)
    column = rng.integers(0, rows - 1, num_nonzero)

    return (row, column)


def merge_isolates(graph: rx.PyDiGraph):
    isolates = rx.isolates(graph)
    for isolate in isolates[1:]:
        graph.merge_nodes(isolate, isolates[0])


def _bisimulation_from_matrix(coordinates):
    """
    Computes the bisimulation from the given adjacency matrix.
    """
    t = perf_counter()
    nodes = set(coordinates[0]).union(set(coordinates[1]))
    graph = rx.PyDiGraph()
    graph.add_nodes_from(list(nodes))
    graph.add_edges_from_no_data(
        [arrow for arrow in zip(coordinates[0], coordinates[1])]
    )

    t = perf_counter()
    bisim = rx.digraph_maximum_bisimulation(graph)
    return bisim


_generate_sparse_adjacency_matrix = (
    lambda nodes, sparsity: generate_binary_sparse_matrix(nodes, nodes, 1 - sparsity)
)


def _run_experiment(nodes: int, sparsity):
    """
    Runs the experiment for the given parameters.
    """

    adjacency_matrix = _generate_sparse_adjacency_matrix(nodes, sparsity)
    bisimulation = _bisimulation_from_matrix(adjacency_matrix)

    return coo_matrix(
        (
            [len(bisimulation) for _ in range(len(adjacency_matrix[0]))],
            adjacency_matrix,
        ),
        shape=(nodes, nodes),
    ).toarray()


def execute_experiment(node_count, sp, sample_rate):
    collector = np.zeros((node_count, node_count))
    for _ in tqdm(range(sample_rate)):
        collector += _run_experiment(node_count, sp)

    return collector


def main():
    instances = 112
    # for node_count, sp in zip(nodes, sparsity):
    #     print(f"Running experiment for {node_count} nodes and sparsity {sp} with sample rate {sample_rate[node_count]}")
    #     experiment_results[(node_count, sp)] = [
    #         _run_experiment(node_count, sp) for _ in tqdm(range(sample_rate[node_count]), mininterval=0.1)
    #     ]

    experiment_results = defaultdict(list)

    with concurrent.futures.ProcessPoolExecutor(max_workers=15) as executor:
        futures = []
        for _ in range(instances):
            future = executor.submit(execute_experiment, 1300, 0.995, 1000)
            futures.append(future)

        # Wait for all futures to complete
        def wait_for(futures):
            index = 0
            for future in concurrent.futures.as_completed(futures):
                yield index, future
                index += 1

        try:
            for future in tqdm(wait_for(futures), total=len(futures), miniters=10):
                res = future[1].result()
                experiment_results[list(res.keys())[0]].extend(list(res.values())[0])
        except:
            pass

    with open(f"experiment_results_{time.time()}.pickle", "wb") as file:
        pickle.dump(experiment_results, file)


if __name__ == "__main__":
    main()
