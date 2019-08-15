from isorank import RWR, IsoRank
from scipy import sparse
import numpy as np
import pickle
import sys


def load_adj(fname_pckl, net_type='experimental'):
    """Load pickle file with string networks."""
    net = pickle.load(open(fname_pckl, 'rb'))
    proteins = net['prot_IDs']
    A = net['nets'][net_type]
    A = A/A.max()
    prot2index = {}
    ii = 0
    for prot in proteins:
        prot2index[prot] = ii
        ii += 1
    return prot2index, A, proteins


def load_blastp(fname, prot2index_1, prot2index_2):
    R_12 = sparse.lil_matrix((len(prot2index_1), len(prot2index_2)))
    fRead = open(fname, 'r')
    for line in fRead:
        splitted = line.strip().split('\t')
        prot1 = splitted[0]
        prot2 = splitted[1]
        e_val = float(splitted[10])
        if e_val == 0:
            e_val = 180.0
        else:
            e_val = -1.0*np.log10(e_val)
        if prot1 in prot2index_1 and prot2 in prot2index_2:
            ii = prot2index_1[prot1]
            jj = prot2index_2[prot2]
            R_12[ii, jj] = e_val/180.0
    fRead.close()

    return R_12


def save_rwr_matrices(tax_ids, network_folder='./network_files/'):
    for ii in range(0, len(tax_ids)):
        net = {}
        prot2index, A, net_prots = load_adj(network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl")
        net['net'] = RWR(A, maxiter=4)
        net['prot_IDs'] = net_prots
        pickle.dump(net, open(network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "wb"))
        print ('\n')

def save_block_matrices(alpha, tax_ids, network_folder='./network_files/', blast_folder='./blast_files/', block_matrix_folder='./block_matrix_files/'):
    for ii in range(0, len(tax_ids)):
        prot2index_1, A_1, _ = load_adj(network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl")
        for jj in range(ii+1, len(tax_ids)):
            prot2index_2, A_2, _ = load_adj(network_folder + tax_ids[jj] + "_networks_string.v10.5.pckl")
            R = load_blastp(blast_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_blastp.tab", prot2index_1, prot2index_2)
            # ***adding this normalization:
            R /= R.sum()
            R = IsoRank(A_1, A_2, R, alpha=alpha, maxiter=4)
                
            pickle.dump(R, open(block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "wb"))


if __name__ == "__main__":
    # Eukaryotes:
    # python create_block_matrix.py 0.6 4896,4932,9606,10090,7227,3702,6239,10116 ./ ../annot/string_annot/
    alpha = float(sys.argv[1])
    # tax_ids_2 = ['511145', '7227', '10090', '6239', '4932', '9606']
    # tax_ids_1 = ['553174']
    # tax_ids = ['155864', '199310', '316385', '316407', '511145', '220664', '208964', '553174']
    tax_ids = sys.argv[2].split(',')
    network_folder = sys.argv[3]
    blast_folder = sys.argv[4]
    block_matrix_folder = sys.argv[5]
    if network_folder[-1] != '/':
        network_folder = network_folder + '/'
    if blast_folder[-1] != '/':
        blast_folder = blast_folder + '/'
    if block_matrix_folder[-1] != '/':
        block_matrix_folder = block_matrix_folder + '/'
    save_rwr_matrices(tax_ids, network_folder=network_folder)
    save_block_matrices(alpha, tax_ids, network_folder=network_folder, blast_folder=blast_folder, block_matrix_folder=block_matrix_folder)
