from scipy import sparse
import numpy as np
import pickle
import csv
import obonet
import sys
from os.path import isfile
import os
import subprocess

# Edited for string v11 instead of 10.5

def process_version_11_annot_file(graph, root_terms, go_path, annot_folder, min_coverage, max_coverage):

    Annot = {}
    Annot['prot_IDs'] = []
    Annot['go_IDs'] = {}
    Annot['go_IDs']['molecular_function'] = []
    Annot['go_IDs']['biological_process'] = []
    Annot['go_IDs']['cellular_component'] = []
    Annot['annot'] = {}

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

    with open(go_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader) # skip header for version 11
        for row in reader:
            '''
            tax_id = row[0].strip()
            string_id = row[1].strip()
            protein_name = row[2].strip()
            go_id = row[3].strip()
            go_name = row[4].strip()
            evidence = row[6].strip()
            '''
            # no evidence, protein name or go name in version 11
            # only string id, tax id, and go id, and ontology
            tax_id = row[0].strip()
            go_id = row[2].strip()
            string_id = row[3].strip()
            
            if tax_id in tax_ids and go_id in graph and go_id not in root_terms:
                namespace = graph.nodes[go_id]['namespace']
                lines.append([namespace, string_id, go_id])
                if string_id not in string2idx:
                    string2idx[string_id] = ii
                    Annot['prot_IDs'].append(string_id)
                    ii += 1
                if go_id not in go2idx[namespace]:
                    go2idx[namespace][go_id] = jj[namespace]
                    Annot['go_IDs'][namespace].append(go_id)
                    jj[namespace] += 1
    Annot['annot']['molecular_function'] = sparse.lil_matrix((ii, jj['molecular_function']))
    Annot['annot']['biological_process'] = sparse.lil_matrix((ii, jj['biological_process']))
    Annot['annot']['cellular_component'] = sparse.lil_matrix((ii, jj['cellular_component']))

    for entry in lines:
        namespace = entry[0]
        string_id = entry[1]
        go_id = entry[2]
        Annot['annot'][namespace][string2idx[string_id], go2idx[namespace][go_id]] = 1.0

        
    name_prefix = '-'.join(tax_ids)
    pickle.dump(Annot, open(annot_folder + name_prefix + '_string.01_2019_annotations.pckl', 'wb'))
    for ont in ['molecular_function', 'biological_process', 'cellular_component']:
        num_prots = Annot['annot'][ont].shape[0]
        go_term_coverages = np.sum(Annot['annot'][ont], axis=0)/num_prots
        terms_are_covered = np.logical_and((go_term_coverages > min_coverage), (go_term_coverages < max_coverage))
        chosen_go_inds = np.where(terms_are_covered)[1]
        chosen_go_IDs = np.array(Annot['go_IDs'][ont])[chosen_go_inds] 
        print(chosen_go_IDs)
        print(len(chosen_go_IDs))
        pickle.dump(chosen_go_IDs, open(annot_folder + name_prefix + '_' + ont + '_train_goids.pckl', 'wb'))


def process_version_10_annot_file(graph, root_terms, evidence_codes, go_path, annot_folder, min_coverage, max_coverage):

    Annot = {}
    Annot['prot_IDs'] = []
    Annot['go_IDs'] = {}
    Annot['go_IDs']['molecular_function'] = []
    Annot['go_IDs']['biological_process'] = []
    Annot['go_IDs']['cellular_component'] = []
    Annot['go_names'] = {}
    Annot['prot_names'] = []
    Annot['go_names']['molecular_function'] = []
    Annot['go_names']['biological_process'] = []
    Annot['go_names']['cellular_component'] = []
    Annot['annot'] = {}

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
    with open(go_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            tax_id = row[0].strip()
            string_id = row[1].strip()
            protein_name = row[2].strip()
            go_id = row[3].strip()
            go_name = row[4].strip()
            evidence = row[6].strip()
            
            if evidence_codes == 'all': 
                if tax_id in tax_ids and go_id in graph and go_id not in root_terms:
                    string_id = tax_id + '.' + string_id
                    namespace = graph.nodes[go_id]['namespace']
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
            else: # filter by evidence code
                if tax_id in tax_ids and evidence in evidence_codes and go_id in graph and go_id not in root_terms:
                    string_id = tax_id + '.' + string_id
                    namespace = graph.nodes[go_id]['namespace']
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
    
    Annot['annot']['molecular_function'] = sparse.lil_matrix((ii, jj['molecular_function']))
    Annot['annot']['biological_process'] = sparse.lil_matrix((ii, jj['biological_process']))
    Annot['annot']['cellular_component'] = sparse.lil_matrix((ii, jj['cellular_component']))

    for entry in lines:
        namespace = entry[0]
        string_id = entry[1]
        go_id = entry[2]
        Annot['annot'][namespace][string2idx[string_id], go2idx[namespace][go_id]] = 1.0
        
    name_prefix = '-'.join(tax_ids)
    pickle.dump(Annot, open(annot_folder + name_prefix + '_string.04_2015_annotations.pckl', 'wb'))
    for ont in ['molecular_function', 'biological_process', 'cellular_component']:
        num_prots = Annot['annot'][ont].shape[0]
        go_term_coverages = np.sum(Annot['annot'][ont], axis=0)/num_prots
        terms_are_covered = np.logical_and((go_term_coverages > min_coverage), (go_term_coverages < max_coverage))
        chosen_go_inds = np.where(terms_are_covered)[1]
        chosen_go_IDs = np.array(Annot['go_IDs'][ont])[chosen_go_inds] 
        print(chosen_go_IDs)
        print(len(chosen_go_IDs))
        pickle.dump(chosen_go_IDs, open(annot_folder + name_prefix + '_' + ont + '_train_goids.pckl', 'wb'))
        chosen_go_names = np.array(Annot['go_names'][ont])[chosen_go_inds] 
        with open(annot_folder + name_prefix + '_' + ont + '_train_gonames.txt', 'w') as f:
            for i, go_name in enumerate(chosen_go_names):
                f.write("%s\t%s\n" % (chosen_go_IDs[i], go_name))


def save_annots(tax_ids, annot_folder='./string_annot/', min_coverage=0.005, max_coverage=0.05, version='11', evidence='all'):
    # read *.obo file
    try:
        graph = obonet.read_obo(open('./go-basic.obo', 'r'))
    except FileNotFoundError:
        os.system('wget http://purl.obolibrary.org/obo/go/go-basic.obo')
        graph = obonet.read_obo(open('./go-basic.obo', 'r'))

    root_terms = ['GO:0003674', 'GO:0008150', 'GO:0005575']
    if evidence == 'all':
        evidence_codes = 'all'
    elif evidence == 'experimental':
        evidence_codes = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED']
    else:
        print('Evidence setting must be either \'all\' or \'experimental\'. Exit.')
        exit()

    if version == '11':
        go_path = annot_folder + 'all_organisms.GO_2_string.2018.tsv'
    elif version == '10':
        go_path = annot_folder + 'all_go_knowledge_full.tsv'
    if not isfile(go_path):
        print(go_path + ' not found. Downloading and gunzipping it.')
        if version == '11':
            subprocess.run(['wget', '-P', annot_folder, 'https://string-db.org/mapping_files/geneontology/all_organisms.GO_2_string.2018.tsv.gz'])
        elif version == '10':
            subprocess.run(['wget', '-P', annot_folder, 'http://version10.string-db.org/mapping_files/gene_ontology_mappings/all_go_knowledge_full.tsv.gz'])
        else:
            print('Wrong version! Choose \'10\' or \'11\'.')
            exit()
        subprocess.run(['gunzip', '-f', go_path + '.gz'])

    if version == '11':
        process_version_11_annot_file(graph, root_terms, go_path, annot_folder, min_coverage, max_coverage)
    elif version == '10':
        process_version_10_annot_file(graph, root_terms, evidence_codes, go_path, annot_folder, min_coverage, max_coverage)

    

if __name__ == '__main__':
    tax_ids = sys.argv[1].split(',')
    annot_folder = sys.argv[2]
    if annot_folder[-1] != '/':
        annot_folder += '/'
    if len(sys.argv) > 3:
        string_version = sys.argv[3]
    else:
        string_version = '11'
    save_annots(tax_ids, annot_folder=annot_folder, version=string_version)

