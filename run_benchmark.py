from statistics import mean
from algorithm_aco import ACO
from common import create_weighted_matrix, get_error_rate, merge_result_oligos
from instances import generate_full_instance_for_ants_tests, generate_problem_instance
import numpy as np


def main():
    spectrum = generate_full_instance_for_ants_tests(n=200, k=9)

    oligos = spectrum.get_oligos_as_strings()

    weights = create_weighted_matrix(oligos)
    with np.printoptions(threshold=np.inf):
          print(weights)

    print("Weight = 1 count: {}".format(np.count_nonzero(weights == 1)))

    aco = ACO(nodes=oligos, distances=weights)
    aco.run()

    result_str, result_oligos = merge_result_oligos(
        path=aco.best_path,
        oligos=oligos,
        weights=weights
    )

    print("EXPECTED: {}".format([o.name for o in spectrum.oligos_list]))
    print("ACTUAL  : {}".format([oligos[n] for n in aco.best_path]))
    print("JOIN : {}".format(result_oligos))

    print("ORIGINAL: {}".format(spectrum.original_dna))
    print("RESULT  : {}".format(result_str))
    print(get_error_rate(result_str, spectrum.original_dna))


def benchmark_ants_count():
    n = 500
    k = 9

    with open('input/dna.txt', 'r') as f:
        sequences = f.readlines()

    results = {}

    for seq in sequences[0:10]:
        negative_errors = 0
        positive_errors = 0

        spectrum = generate_problem_instance(
            dna_sequence=seq,
            dna_length=n,
            oligos_length=k,
            negative_errors_percent=negative_errors,
            positive_errors_percent=positive_errors
        )

        oligos = spectrum.get_oligos_as_strings()

        weights = create_weighted_matrix(oligos)

        for ants in [50, 100, 150, 200, 250]:
            if not ants in results:
                results[ants] = []

            aco = ACO(nodes=oligos, distances=weights,
                      ants_to_nodes_ratio=ants/n)
            aco.run()

            result_str, _ = merge_result_oligos(
                path=aco.best_path,
                oligos=oligos,
                weights=weights
            )

            error_rate = get_error_rate(result_str, spectrum.original_dna)
            print("{}: {}".format(ants, error_rate))

            results[ants].append(error_rate)

    for k in results:
        print("{}, {}".format(k, mean(results[k])))


def benchmark_evaporation():
    n = 500
    k = 9

    with open('input/dna.txt', 'r') as f:
        sequences = f.readlines()

    results = {}

    for seq in sequences[0:5]:
        negative_errors = 2
        positive_errors = 2

        spectrum = generate_problem_instance(
            dna_sequence=seq,
            dna_length=n,
            oligos_length=k,
            negative_errors_percent=negative_errors,
            positive_errors_percent=positive_errors
        )

        oligos = spectrum.get_oligos_as_strings()

        weights = create_weighted_matrix(oligos)

        for ev in [0.01, 0.05, 0.1, 0.2, 0.3]:
            if not ev in results:
                results[ev] = []

            aco = ACO(nodes=oligos, distances=weights, evaporation=ev)
            aco.run()

            result_str, _ = merge_result_oligos(
                path=aco.best_path,
                oligos=oligos,
                weights=weights
            )

            error_rate = get_error_rate(result_str, spectrum.original_dna)
            print("{}: {}".format(ev, error_rate))

            results[ev].append(error_rate)

    for k in results:
        print("{}, {}".format(k, mean(results[k])))


def benchmark_negative_errors():
    n = 500
    k = 7

    with open('input/dna.txt', 'r') as f:
        sequences = f.readlines()

    results = {}

    for seq in sequences[0:5]:
        for ev in [2, 4, 6, 8, 10]:
            negative_errors = ev
            positive_errors = 2

            spectrum = generate_problem_instance(
                dna_sequence=seq,
                dna_length=n,
                oligos_length=k,
                negative_errors_percent=negative_errors,
                positive_errors_percent=positive_errors
            )

            oligos = spectrum.get_oligos_as_strings()

            weights = create_weighted_matrix(oligos)

            if not ev in results:
                results[ev] = []

            aco = ACO(nodes=oligos, distances=weights,
                      evaporation=0.3, ants_to_nodes_ratio=0.1)
            aco.run()

            result_str, _ = merge_result_oligos(
                path=aco.best_path,
                oligos=oligos,
                weights=weights
            )

            error_rate = get_error_rate(result_str, spectrum.original_dna)
            print("{}: {}".format(ev, error_rate))

            results[ev].append(error_rate)

    for k in results:
        print("{}, {}".format(k, mean(results[k])))


def benchmark_k():
    n = 500

    with open('input/dna.txt', 'r') as f:
        sequences = f.readlines()

    results = {}

    for seq in sequences[0:5]:
        for ev in [6, 7, 8, 9, 10]:
            negative_errors = 2
            positive_errors = 2

            spectrum = generate_problem_instance(
                dna_sequence=seq,
                dna_length=n,
                oligos_length=ev,
                negative_errors_percent=negative_errors,
                positive_errors_percent=positive_errors
            )

            oligos = spectrum.get_oligos_as_strings()

            weights = create_weighted_matrix(oligos)

            if not ev in results:
                results[ev] = []

            aco = ACO(nodes=oligos, distances=weights,
                      evaporation=0.3, ants_to_nodes_ratio=0.1)
            aco.run()

            result_str, _ = merge_result_oligos(
                path=aco.best_path,
                oligos=oligos,
                weights=weights
            )

            error_rate = get_error_rate(result_str, spectrum.original_dna)
            print("{}: {}".format(ev, error_rate))

            results[ev].append(error_rate)

    for k in results:
        print("{}, {}".format(k, mean(results[k])))


benchmark_k()
