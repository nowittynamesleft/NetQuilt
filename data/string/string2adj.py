import networkx as nx
import pickle
import numpy as np
import subprocess
import sys


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


def load_string_nets(read_fname):
    net_names = ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 'experimental', 'database', 'textmining', 'combined']
    graphs = []
    for name in net_names:
        graphs.append(nx.Graph(name=name))
    fRead = open(read_fname, 'r')
    fRead.readline()
    for line in fRead:
        splitted = line.strip().split()
        prot1 = str(splitted[0])
        # prot1 = prot1.split('.')[1]
        prot2 = str(splitted[1])
        # prot2 = prot2.split('.')[1]
        scores = splitted[2:]
        scores = [float(s) for s in scores]
        for ii in range(0, len(scores)):
            if not graphs[ii].has_node(prot1):
                graphs[ii].add_node(prot1)
            if not graphs[ii].has_node(prot2):
                graphs[ii].add_node(prot2)
        for ii in range(0, len(scores)):
            if scores[ii] > 0:
                graphs[ii].add_edge(prot1, prot2, weight=float(scores[ii]))
    fRead.close()
    # for ii in range(0, len(graphs)):
    #    graphs[ii] = nx.relabel_nodes(graphs[ii], string2uniprot)
    String = {}
    String['prot_IDs'] = list(graphs[0].nodes())
    String['nets'] = {}
    for ii in range(0, len(graphs)):
        String['nets'][net_names[ii]] = nx.adjacency_matrix(graphs[ii], nodelist=String['prot_IDs'])
        print (net_names[ii], graphs[ii].order(), graphs[ii].size())

    return String

def save_networks(tax_ids, network_folder='./network_files/'):
    for tax in tax_ids:
        print (tax)
        fname = tax + '.protein.links.detailed.v10.5.txt.gz'
        try:
            String = load_string_nets(network_folder + fname[:-3])
        except FileNotFoundError:
            subprocess.run(['wget', '-P', network_folder, 'https://version-10-5.string-db.org/download/protein.links.detailed.v10.5/' + fname])
            subprocess.run(['gunzip', '-f', network_folder + fname])
        String = load_string_nets(network_folder + fname[:-3])
        pickle.dump(String, open(network_folder + tax + "_networks_string.v10.5.pckl", "wb"))
        #String = pickle.load(open("./prevotella_melaninogenica/" + tax + "_networks_string.v10.5.pckl", "rb"))
        #net = String['nets']['experimental'].todense()
        #print (tax,  net.shape[0], np.count_nonzero(net)/2)

if __name__ == "__main__":
    #tax_ids = ['199310', '155864', '511145', '316407', '316385', '220664', '208964', '553174']
    tax_ids = sys.argv[1].split(',')
    save_networks(tax_ids)
