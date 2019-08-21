from scipy import sparse
import numpy as np
import pickle
import csv
import obonet
import sys
from pathlib import Path

def save_annots(tax_ids):
    # read *.obo file
    graph = obonet.read_obo(open('./go-basic.obo', 'r'))

    root_terms = ['GO:0003674', 'GO:0008150', 'GO:0005575']
    evidence_codes = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED']

    # tax_ids = ['553174']
    #tax_ids = ['9606']
    # tax_ids = ['9606', '4932']
    # tax_ids = ['9606', '4932', '6239', '10090', '7227', '511145']
    #tax_ids = ['199310', '155864', '511145', '316407', '316385', '220664', '208964']

    Annot = {}
    Annot['prot_IDs'] = []
    Annot['prot_names'] = []

    Annot['go_IDs'] = {}
    Annot['go_IDs']['molecular_function'] = []
    Annot['go_IDs']['biological_process'] = []
    Annot['go_IDs']['cellular_component'] = []

    Annot['go_names'] = {}
    Annot['go_names']['molecular_function'] = []
    Annot['go_names']['biological_process'] = []
    Annot['go_names']['cellular_component'] = []


    ii = 0

    string2idx = {}
    go2idx = {}
    go2idx['molecular_function'] = {}
    go2idx['biological_process'] = {}
    go2idx['cellular_component'] = {}

    jj = {}
    jj['molecular_function'] = 0
    jj['biological_process'] = 0
    jj['cellular_component'] = 0

    lines = []
    annot_folder = './string_annot/'
    go_path = annot_folder + 'all_go_knowledge_full.tsv'
    if not Path(go_path):
        print(go_path + ' not found. Downloading and gunzipping it.')
        subprocess.run(['wget', '-P', annot_folder, 'https://version-10-5.string-db.org/mapping_files/gene_ontology_mappings/all_go_knowledge_full.tsv.gz'])
        subprocess.run(['gunzip', '-f', go_path + '.gz'])

    with open('string_annot/all_go_knowledge_full.tsv') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            tax_id = row[0].strip()
            string_id = row[1].strip()
            protein_name = row[2].strip()
            go_id = row[3].strip()
            go_name = row[4].strip()
            evidence = row[6].strip()
            if tax_id in tax_ids and evidence in evidence_codes and go_id in graph and go_id not in root_terms:
                string_id = tax_id + '.' + string_id
                namespace = graph.node[go_id]['namespace']
                lines.append([namespace, string_id, go_id])
                if string_id not in string2idx:
                    string2idx[string_id] = ii
                    Annot['prot_IDs'].append(string_id)
                    Annot['prot_names'].append(protein_name)
                    ii += 1
                if go_id not in go2idx[namespace]:
                    go2idx[namespace][go_id] = jj[namespace]
                    Annot['go_IDs'][namespace].append(go_id)
                    Annot['go_names'][namespace].append(go_name)
                    jj[namespace] += 1


    Annot['annot'] = {}
    Annot['annot']['molecular_function'] = sparse.lil_matrix((ii, jj['molecular_function']))
    Annot['annot']['biological_process'] = sparse.lil_matrix((ii, jj['biological_process']))
    Annot['annot']['cellular_component'] = sparse.lil_matrix((ii, jj['cellular_component']))
        

    for entry in lines:
        namespace = entry[0]
        string_id = entry[1]
        go_id = entry[2]
        Annot['annot'][namespace][string2idx[string_id], go2idx[namespace][go_id]] = 1.0

    # pickle.dump(Annot, open('9606-4932_string.04_2015_annotations.pckl', 'wb'))
    # pickle.dump(Annot, open('model_orgs_string.04_2015_annotations.pckl', 'wb'))
    # pickle.dump(Annot, open('9606_string.04_2015_annotations.pckl', 'wb'))
    # pickle.dump(Annot, open('553174_string.04_2015_annotations.pckl', 'wb'))
    name_prefix = '-'.join(tax_ids)
    pickle.dump(Annot, open('./string_annot/' + name_prefix + '_string.04_2015_annotations.pckl', 'wb'))
    min_coverage = 0.01
    max_coverage = 0.05
    for ont in ['molecular_function', 'biological_process', 'cellular_component']:
        num_prots = Annot['annot'][ont].shape[0]
        go_term_coverages = np.sum(Annot['annot'][ont], axis=0)/num_prots
        terms_are_covered = np.logical_and((go_term_coverages > min_coverage), (go_term_coverages < max_coverage))
        chosen_go_inds = np.where(terms_are_covered)[1]
        chosen_go_IDs = np.array(Annot['go_IDs'][ont])[chosen_go_inds] 
        print(chosen_go_IDs)
        print(len(chosen_go_IDs))
        pickle.dump(chosen_go_IDs, open('./string_annot/' + name_prefix + '_' + ont + '_train_goids.pckl', 'wb'))
        chosen_go_names = np.array(Annot['go_names'][ont])[chosen_go_inds] 
        with open('./string_annot/' + name_prefix + '_' + ont + '_train_gonames.txt', 'w') as f:
            for i, go_name in enumerate(chosen_go_names):
                f.write("%s\t%s\n" % (chosen_go_IDs[i], go_name))
    

if __name__ == '__main__':
    tax_ids = sys.argv[1].split(',')
    save_annots(tax_ids)

