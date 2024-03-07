from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from random import choices
from typing import List
from uuid import uuid4, UUID
import numpy as np
from numpy import ndarray

from common import NUCLEOTIDES


@dataclass
class Oligonucleotide:
    name: str
    id: UUID
    neighbours: dict = field(default_factory=dict)

    def add_neighbour(self, oligo: Oligonucleotide, offset: int):
        if offset > 4:
            return
        self.neighbours[offset] = oligo

    def define_neighb_level(self, oligo: Oligonucleotide):
        for i in range(1, 5):
            if self.name[i:] == oligo.name[:-i]:
                level = i

        level = len(oligo.name)
        return level

    def __repr__(self):
        return f'Oligo(name: {self.name})'


@dataclass
class Spectrum:
    original_dna: str
    oligos_length: int
    positive_errors: int = 0
    negative_errors: int = 0
    oligos_list: List[Oligonucleotide] = field(default_factory=list)
    duplicates: List[str] = field(default_factory=list)
    matrix: ndarray = field(default_factory=list)

    def __post_init__(self):
        self.oligos_list_copy = self.oligos_list[::]
        self.first_oligo = self.oligos_list[0]

    @property
    def sorted_oligo_list(self):
        return [self.first_oligo] + sorted(self.oligos_list[1:], key=lambda oligo: oligo.name)

    @property
    def oligo_names_list(self):
        return [elem.name for elem in self.sorted_oligo_list]

    def add_to_spectrum(self, oligo: Oligonucleotide):
        self.oligos_list.append(oligo)

    def delete_from_spectrum(self, oligo: Oligonucleotide):
        self.oligos_list.remove(oligo)

    def eliminate_duplicates(self):
        unique_oligos_names = set()
        for oligo in self.sorted_oligo_list:
            if oligo.name not in unique_oligos_names:
                unique_oligos_names.add(oligo.name)
            else:
                self.duplicates.append(oligo.name)
                self.delete_from_spectrum(oligo)

        self.negative_errors = len(self.duplicates)
        print(
            f'Number of negative errors before adding: {self.negative_errors}')

    def include_negative_errors(self, percent: int):
        total_negative_errors = ceil(len(self.oligos_list) * percent / 100)
        if len(self.duplicates) >= total_negative_errors:
            return
        self.negative_errors = total_negative_errors
        print(f'Total number of negative errors: {self.negative_errors}')

        num_oligos_to_remove = self.negative_errors - len(self.duplicates)
        for _ in range(num_oligos_to_remove):
            for oligo in self.sorted_oligo_list[1:]:
                if oligo.name not in self.duplicates:
                    self.delete_from_spectrum(oligo)
                    break

    def generate_new_oligo(self):
        return ''.join(choices(NUCLEOTIDES, k=self.oligos_length))

    def add_false_oligo(self):
        while True:
            print("While loop - generating false oligo...")
            false_oligo_name = self.generate_new_oligo()
            if false_oligo_name not in self.oligo_names_list:
                self.oligos_list.append(
                    Oligonucleotide(false_oligo_name, uuid4()))
                return

    def include_positive_errors(self, percent: int):
        self.positive_errors = ceil(len(self.oligos_list) * percent / 100)
        print(f'Number of positive errors: {self.positive_errors}')
        for _ in range(self.positive_errors):
            self.add_false_oligo()

    def get_matrix_of_connections(self):
        self.matrix = np.zeros(len(self.oligos_list), len(
            self.oligos_list))

        for i in range(len(self.oligos_list)):
            for j in range(len(self.oligos_list)):
                self.matrix[i][j] = self.sorted_oligo_list[i].define_neighb_level(
                    self.sorted_oligo_list[j])
                self.sorted_oligo_list[i].add_neighbour(
                    self.sorted_oligo_list[j], self.matrix[i][j])

    def get_oligos_as_strings(self):
        return [oligo.name for oligo in self.sorted_oligo_list]

    def __str__(self):
        return f'Spectrum(original_dna: {self.original_dna}, oligos_list: {self.sorted_oligo_list})'


def generate_basic_dna(length: int, file_name: str = ""):
    dna_sequence_list = choices(NUCLEOTIDES, k=length)
    dna_sequence = ''.join(dna_sequence_list)

    if not file_name:
        return dna_sequence

    with open(file_name, 'a') as f:
        f.write(dna_sequence + '\n')


def generate_oligonucleotides_spectrum(n: int, k: int, sequence: str = "") -> Spectrum:
    if not sequence:
        original_dna = generate_basic_dna(n)
    else:
        original_dna = sequence

    num_oligos = n - k + 1
    dna_spectrum = Spectrum(
        original_dna=original_dna,
        oligos_length=k,
        oligos_list=[
            Oligonucleotide(original_dna[i:i+k], uuid4())
            for i in range(num_oligos)
        ]
    )
    dna_spectrum.eliminate_duplicates()
    return dna_spectrum


def generate_problem_instance(dna_length: int, oligos_length: int,
                              dna_sequence: str = "",
                              negative_errors_percent: int = 0,
                              positive_errors_percent: int = 0) -> Spectrum:

    dna_spectrum = generate_oligonucleotides_spectrum(
        sequence=dna_sequence, n=dna_length, k=oligos_length)
    dna_spectrum.include_negative_errors(negative_errors_percent)
    dna_spectrum.include_positive_errors(positive_errors_percent)
    return dna_spectrum


def generate_full_instance_for_ants_tests(n, k):
    print("Ants algorithm tests running...")

    with open('input/dna.txt', 'r') as f:
        dna_seq_list = f.readlines()

    negative_errors = 0
    positive_errors = 0

    dna_spectrum = generate_problem_instance(
        dna_sequence=dna_seq_list[6].strip()[:n],
        dna_length=n,
        oligos_length=k,
        negative_errors_percent=negative_errors,
        positive_errors_percent=positive_errors
    )

    return dna_spectrum
