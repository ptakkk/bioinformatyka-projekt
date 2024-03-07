from __future__ import annotations
import numpy as np

NUCLEOTIDES = ['A', 'G', 'C', 'T']


def levenshtein_distance(a: str, b: str):
    m, n = len(a), len(b)
    out = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        out[i][0] = i
    for j in range(n + 1):
        out[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            out[i][j] = min(
                out[i - 1][j] + 1,
                out[i][j - 1] + 1,
                out[i - 1][j - 1] + cost
            )

    distance = out[m][n]

    return distance


def get_error_rate(result: str, original: str):
    distance = levenshtein_distance(result, original)

    return distance / len(original)


def get_shift_between_oligos(a: str, b: str):
    max_len = len(a) - 1

    for i in range(1, max_len):
        if a[i:] == b[:-i]:
            return i

    return max_len


def create_weighted_matrix(nodes: list[str]):
    return np.array([[
        -1 if a == b else get_shift_between_oligos(a, b)
        for b in nodes
    ] for a in nodes])


def merge_result_oligos(path, oligos, weights):
    result_oligos = [oligos[0]]

    for i in range(1, len(path)):
        a = path[i-1]
        b = path[i]

        oligo = oligos[b]

        dist = weights[a][b]
        oligo_part = oligo[-dist:]

        result_oligos.append(oligo_part)

    return ''.join(result_oligos), result_oligos
