import numpy as np
from scipy.sparse import coo_array
import rustworkx as rx
from tqdm import tqdm
from typing import Dict, Tuple, Any, List
import pickle
from itertools import product
import concurrent.futures
import time
from collections import defaultdict


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

    coordinates = np.unravel_index(
        rng.choice(rows**2, num_nonzero, replace=False, shuffle=False).astype(
            np.uint64
        ),
        (rows, cols),
    )
    coordinates = coordinates[0].astype(np.uint64), coordinates[1].astype(np.uint64)

    return coordinates


def _generate_sparse_adjacency_matrix(nodes, sparsity):
    """
    Generates a sparse adjacency matrix with the given number of nodes and sparsity.
    """
    return generate_binary_sparse_matrix(nodes, nodes, density=1 - sparsity)


def merge_isolates(graph: rx.PyDiGraph):
    isolates = rx.isolates(graph)
    for isolate in isolates[1:]:
        graph.merge_nodes(isolate, isolates[0])


def _bisimulation_from_matrix(adjacency_matrix, nodes):
    """
    Computes the bisimulation from the given adjacency matrix.
    """

    graph = rx.PyDiGraph.from_sparse_adjacency_matrix(
        adjacency_matrix[0], adjacency_matrix[1], nodes
    )

    bisimulation = rx.digraph_maximum_bisimulation(graph)

    return bisimulation


def _run_experiment(nodes, sparsity):
    """
    Runs the experiment for the given parameters.
    """

    adjacency_matrix = _generate_sparse_adjacency_matrix(nodes, sparsity)
    bisimulation = _bisimulation_from_matrix(adjacency_matrix, nodes)

    return len(bisimulation)


def execute_experiment(node_count, sp, sample_rate):
    experiment_results = defaultdict(list)
    experiment_results[(node_count, sp)] = [
        _run_experiment(node_count, sp) for _ in range(sample_rate[node_count])
    ]
    return experiment_results


def main():
    nodes = [n for n in range(0, 5000, 50)]
    sparsity = [0.995 + 0.001 * n for n in range(4)]
    sample_rate = {n: 1000 for n in nodes}

    # for node_count, sp in zip(nodes, sparsity):
    #     print(f"Running experiment for {node_count} nodes and sparsity {sp} with sample rate {sample_rate[node_count]}")
    #     experiment_results[(node_count, sp)] = [
    #         _run_experiment(node_count, sp) for _ in tqdm(range(sample_rate[node_count]), mininterval=0.1)
    #     ]

    experiment_results = defaultdict(list)
    #     for node_count, sp in tqdm(product(nodes, sparsity)):
    #         execute_experiment(node_count, sp, sample_rate)

    with concurrent.futures.ProcessPoolExecutor(max_workers=112) as executor:
        futures = []
        for node_count, sp in tqdm(product(nodes, sparsity)):
            future = executor.submit(execute_experiment, node_count, sp, sample_rate)
            futures.append(future)

        # Wait for all futures to complete
        def wait_for(futures):
            index = 0
            for future in concurrent.futures.as_completed(futures):
                yield index, future
                index += 1

        try:
            for future in tqdm(wait_for(futures), total=len(futures), smoothing=0):
                res = future[1].result()
                experiment_results[list(res.keys())[0]].extend(list(res.values())[0])
        except:
            pass

    with open(f"experiment_results_{time.time()}.pickle", "wb") as file:
        pickle.dump(experiment_results, file)


if __name__ == "__main__":
    main()
