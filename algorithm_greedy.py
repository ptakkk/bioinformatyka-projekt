from common import get_error_rate


def reconstruct_dna_from_ordered_spectrum(spectrum, offsets_list):
    reconstructed_dna_string = spectrum.first_oligo.name
    for i in range(len(offsets_list)):
        num_of_letters_to_add = spectrum.oligos_length - offsets_list[i]
        reconstructed_dna_string += spectrum.visited_oligos[i +
                                                            1].name[-num_of_letters_to_add:]
    return reconstructed_dna_string


def choose_next_oligo(oligo, spectrum):
    for offset in sorted(oligo.neighbours.keys()):
        for neighbour in oligo.neighbours[offset]:
            if neighbour.id not in spectrum.visited_oligos_ids:
                return neighbour, offset
    return None, 0


def run_random_algorithm(dna_spectrum):
    print("Random algorithm is running...")
    dna_spectrum.visited_oligos.append(dna_spectrum.first_oligo)

    reconstructed_dna_length = dna_spectrum.oligos_length
    offsets_list = []

    while reconstructed_dna_length < len(dna_spectrum.original_dna):
        next_oligo, offset = choose_next_oligo(
            dna_spectrum.visited_oligos[-1], dna_spectrum)
        offsets_list.append(offset)
        dna_spectrum.visited_oligos.append(next_oligo)
        reconstructed_dna_length += offset

    print(dna_spectrum.visited_oligos)
    result_dna = reconstruct_dna_from_ordered_spectrum(
        dna_spectrum, offsets_list)

    error_rate = get_error_rate(result_dna, dna_spectrum.original_dna)
    return error_rate
