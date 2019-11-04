import pickle
import numpy as np
import pandas as pd
import sys
#from Bio import SeqIO


def get_mapping_dict(mapping_fname): 
    mapping_df = pd.read_csv(mapping_fname, sep='\t', names=['UniprotKB_AC', 'ID_type', 'ID'])
    mapping_df = mapping_df[mapping_df.ID_type == 'STRING']
    mapping_dict = pd.Series(mapping_df.ID.values, index=mapping_df.UniprotKB_AC).to_dict()
    return mapping_dict


def uniprot_to_string_mapping(uniprot_prots, mapping_dict):
    string_ids = []
    missing_inds = []
    for i, prot in enumerate(uniprot_prots):
        if prot in mapping_dict:
            string_ids.append(mapping_dict[prot])
        else:
            missing_inds.append(i)
    missing_inds = np.array(missing_inds)
    string_ids = np.array(string_ids)
    return string_ids, missing_inds 


def remove_missing_annot_prots(annot_prots, Y, mapping_dict):
    new_annot_prots, missing_inds = uniprot_to_string_mapping(annot_prots, mapping_dict)
    Y = np.delete(Y, missing_inds, axis=0)
    return new_annot_prots, Y


def get_prot_ids_and_annots(brenda_fnames, org_name):
    prot_ids = []
    labels = []
    for fname in brenda_fnames:
        f = open(fname, 'r')
        curr_file_labels = []
        for line in f:
            fields = line.split('|')
            if len(fields) > 1:
                uniprot_id = fields[0]
                ec = fields[2]
                org = fields[3]
                if org == org_name:
                    label_fields = ec.split('.')
                    general_label = int(label_fields[0][-1])
                    prot_ids.append(uniprot_id)
                    curr_file_labels.append(general_label)
        print(fname)
        print('number of labels in file:')
        print(len(curr_file_labels))
        labels.extend(curr_file_labels)
    print('Total prot_ids found:')
    print(len(prot_ids))
    print('Total labels found:')
    print(len(labels))
    prot_ids = np.array(prot_ids)
    labels = np.array(labels)
    return prot_ids, labels


def main(brenda_fnames, uniprot2string_fname):
    print('Get mapping dict')
    mapping_dict = get_mapping_dict(uniprot2string_fname)
    print('Done.')
    print(brenda_fnames[0])
    # If you change this:
    #species_name = 'Drosophila melanogaster'
    species_name = 'Mus musculus'

    #uniprot_ids, labels = get_prot_ids_and_annots(brenda_fnames, 'Homo sapiens')
    #uniprot_ids, labels = get_prot_ids_and_annots(brenda_fnames, 'Saccharomyces cerevisiae')
    uniprot_ids, labels = get_prot_ids_and_annots(brenda_fnames, species_name)
    mapped_prots, mapped_labels = remove_missing_annot_prots(uniprot_ids, labels, mapping_dict)
    print('Total samples mapped to string ids:')
    print(mapped_prots.shape)
    print(mapped_labels.shape)
    # Change this as well:
    brenda_outfile = open('mouse_brenda_string_protein_labels.pckl', 'wb')

    pickle.dump({'prot_IDs': mapped_prots, 'labels': mapped_labels}, brenda_outfile)
    brenda_outfile.close()
     

if __name__ == '__main__':
    brenda_fnames = sys.argv[1:-2]
    uniprot2string_fname = sys.argv[-1]
    main(brenda_fnames, uniprot2string_fname)
