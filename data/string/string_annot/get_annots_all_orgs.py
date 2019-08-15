from scipy import sparse
import numpy as np
import pickle
import csv
import obonet


# read *.obo file
graph = obonet.read_obo(open('../python_th/go-basic.obo', 'r'))

root_terms = ['GO:0003674', 'GO:0008150', 'GO:0005575']
evidence_codes = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC', 'CURATED']

# tax_ids = ['553174']
# tax_ids = ['9606']
# tax_ids = ['9606', '4932']
# tax_ids = ['9606', '4932', '6239', '10090', '7227', '511145']
# tax_ids = ['199310', '155864', '511145', '316407', '316385', '220664', '208964']

Annot = {}


ii = {}

string2idx = {}
go2idx = {}

jj = {}

lines = []
with open('all_go_knowledge_full.tsv') as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    for row in reader:
        tax_id = row[0].strip()
        string_id = row[1].strip()
        protein_name = row[2].strip()
        go_id = row[3].strip()
        go_name = row[4].strip()
        evidence = row[6].strip()
        if evidence in evidence_codes and go_id in graph and go_id not in root_terms: # get all tax ids
            string_id = tax_id + '.' + string_id
            namespace = graph.node[go_id]['namespace']
            lines.append([namespace, string_id, go_id, tax_id])
            if tax_id not in Annot:
                Annot[tax_id] = {}
                ii[tax_id] = 0
                jj[tax_id] = {}
                jj[tax_id]['molecular_function'] = 0
                jj[tax_id]['biological_process'] = 0
                jj[tax_id]['cellular_component'] = 0
                go2idx[tax_id] = {}
                go2idx[tax_id]['molecular_function'] = {}
                go2idx[tax_id]['biological_process'] = {}
                go2idx[tax_id]['cellular_component'] = {}
                string2idx[tax_id] = {}

                Annot[tax_id]['prot_IDs'] = []
                Annot[tax_id]['prot_names'] = []

                Annot[tax_id]['go_IDs'] = {}
                Annot[tax_id]['go_IDs']['molecular_function'] = []
                Annot[tax_id]['go_IDs']['biological_process'] = []
                Annot[tax_id]['go_IDs']['cellular_component'] = []

                Annot[tax_id]['go_names'] = {}
                Annot[tax_id]['go_names']['molecular_function'] = []
                Annot[tax_id]['go_names']['biological_process'] = []
                Annot[tax_id]['go_names']['cellular_component'] = []
            if string_id not in string2idx[tax_id]:
                string2idx[tax_id][string_id] = ii[tax_id]
                Annot[tax_id]['prot_IDs'].append(string_id)
                Annot[tax_id]['prot_names'].append(protein_name)
                ii[tax_id] += 1
            if go_id not in go2idx[tax_id][namespace]:
                go2idx[tax_id][namespace][go_id] = jj[tax_id][namespace]
                Annot[tax_id]['go_IDs'][namespace].append(go_id)
                Annot[tax_id]['go_names'][namespace].append(go_name)
                jj[tax_id][namespace] += 1


print('Adding sparse lil matrices')
for tax_id in Annot.keys():
    Annot[tax_id]['annot'] = {}
    Annot[tax_id]['annot']['molecular_function'] = sparse.lil_matrix((ii[tax_id], jj[tax_id]['molecular_function']))
    Annot[tax_id]['annot']['biological_process'] = sparse.lil_matrix((ii[tax_id], jj[tax_id]['biological_process']))
    Annot[tax_id]['annot']['cellular_component'] = sparse.lil_matrix((ii[tax_id], jj[tax_id]['cellular_component']))

print('Filling sparse lil matrices')
for entry in lines:
    namespace = entry[0]
    string_id = entry[1]
    go_id = entry[2]
    tax_id = entry[3]
    Annot[tax_id]['annot'][namespace][string2idx[tax_id][string_id], go2idx[tax_id][namespace][go_id]] = 1.0

# pickle.dump(Annot, open('9606-4932_string.04_2015_annotations.pckl', 'wb'))
# pickle.dump(Annot, open('model_orgs_string.04_2015_annotations.pckl', 'wb'))
# pickle.dump(Annot, open('9606_string.04_2015_annotations.pckl', 'wb'))
# pickle.dump(Annot, open('553174_string.04_2015_annotations.pckl', 'wb'))
print('Dumping')
pickle.dump(Annot, open('all_string.04_2015_annotations.pckl', 'wb'))
