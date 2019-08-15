import numpy as np
from Bio import SeqIO
import pickle

MAX_SEQ_LEN = 600


def load_mappings(fname):
    # mapping string id to uniprot ids
    string2uniprot = {}
    fRead = open(fname, 'r')
    for line in fRead:
        splitted = line.strip().split()
        uniprot = splitted[0]
        string = splitted[1]
        if string not in string2uniprot:
            string2uniprot[string] = []
        string2uniprot[string].append(uniprot)
    fRead.close()

    for string in string2uniprot:
        string2uniprot[string] = '|'.join(string2uniprot[string])

    return string2uniprot


def get_seq_vecs(sequences, char_indices):
    print('### Generating seq vectors...')
    indexes_seqs = np.zeros((sequences.shape[0], sequences.shape[1]), dtype=np.int)
    for i in range(0, len(sequences)):
        for j in range(0, len(sequences[i])):
            if sequences[i][j] in char_indices:
                indexes_seqs[i][j] = char_indices[sequences[i][j]]
    return indexes_seqs


def process_sequences(entries, maxlen=MAX_SEQ_LEN):
    sequences = []
    proteins = []
    for entry_idx in range(0, len(entries)):
        prot_id = entries[entry_idx][0]
        entry = entries[entry_idx][1]
        entry_chars = list(entry)
        entry_chars = [char for char in entry_chars if char != '\n']
        sequences.append(entry_chars)
        proteins.append(prot_id)

    nb_samples = len(sequences)
    x = np.zeros((nb_samples, maxlen), dtype=np.str)
    for idx, s in enumerate(sequences):
        trunc = np.asarray(s, dtype=np.str)
        if maxlen < len(trunc):
            x[idx] = trunc[:maxlen]
        elif(maxlen > len(trunc)):
            x[idx, 0:len(trunc)] = trunc
        else:
            x[idx] = trunc
    return x, proteins


def get_char_indices():
    # Amino-acid letters
    chars = ['A', 'R', 'N', 'D', 'B', 'C', 'Q', 'E', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    char_indices = dict()
    for idx, char in enumerate(chars):
        char_indices[char] = idx
    return char_indices


def load_FASTA(filename, string2uniprot):
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    entries = [(str(entry.id).split('.')[1], str(entry.seq)) for entry in SeqIO.parse(infile, 'fasta')]
    if(len(entries) == 0):
        return False
    infile.close()

    entries = [(string2uniprot[prot], seq) for (prot, seq) in entries if prot in string2uniprot]

    return entries


if __name__ == "__main__":
    string2uniprot = load_mappings("9606_uniprot_2_string.txt")
    entries = load_FASTA('9606.protein.sequences.v10.5.fa', string2uniprot)
    seq_vec, proteins = process_sequences(entries)
    X = get_seq_vecs(seq_vec, get_char_indices())
    Fasta = {}
    Fasta['prot_IDs'] = proteins
    Fasta['sequences'] = X
    pickle.dump(Fasta, open("9606_sequences_string.v10.5.pckl", "wb"))
