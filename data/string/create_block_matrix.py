from isorank import RWR, IsoRank
from scipy import sparse
from get_fastas_and_blasts import interspecies_blast
from string2adj import save_networks
import numpy as np
import pickle
import sys
from multiprocessing import Pool
import itertools
import multiprocessing
from os import path


# Edited for STRING v11 instead of v10.5

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


def get_single_rwr(tax_id, network_folder):
    net = {}
    #network_file = network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl"
    network_file = network_folder + tax_id + "_networks_string.v11.0.pckl"
    rwr_fname = network_folder + tax_id + "_rwr_features_string.v11.0.pckl"
    if path.exists(rwr_fname):
        print('RWR file ' + rwr_fname + ' already exists; skipping')
        return
    try:
        prot2index, A, net_prots = load_adj(network_file)
    except FileNotFoundError:
        print('Network file ' + network_file + ' not found. Downloading it and then computing rwr.')
        save_networks([tax_id], network_folder=network_folder) # save network that is not found
        prot2index, A, net_prots = load_adj(network_file)
    net['net'] = RWR(A, maxiter=1000)
    net['prot_IDs'] = net_prots
    #pickle.dump(net, open(network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "wb"))
    pickle.dump(net, open(rwr_fname, "wb"), protocol=4)
    print('Saved ' + rwr_fname)
    print ('\n')


def get_single_isorank_block(tax_id_combo, alpha, network_folder, blast_folder, block_matrix_folder, rand_init, ones_init):
    tax_id_1 = tax_id_combo[0]
    tax_id_2 = tax_id_combo[1]

    block_mat_fname = block_matrix_folder + tax_id_1 + "-" + tax_id_2 + "_alpha_" + str(alpha) + "_block_matrix.pckl"
    leaveout_1 = False
    leaveout_2 = False
    if '-leaveout' in tax_id_1:
        tax_id_1 = tax_id_1.split('-')[0] # get actual tax id to load network file for the protein ids, not the adjacency matrix
        leaveout_1 = True
    if '-leaveout' in tax_id_2:
        tax_id_2 = tax_id_2.split('-')[0]
        leaveout_2 = True
    if path.exists(block_mat_fname):
        print('Block mat file ' + block_mat_fname + ' already exists; skipping')
        return
    elif tax_id_1 == tax_id_2:
        print('No isorank matrix will be computed between a species with itself')
        return
    #prot2index_1, A_1, _ = load_adj(network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl")
    #prot2index_2, A_2, _ = load_adj(network_folder + tax_ids[jj] + "_networks_string.v10.5.pckl")
    prot2index_1, A_1, _ = load_adj(network_folder + tax_id_1 + "_networks_string.v11.0.pckl")
    prot2index_2, A_2, _ = load_adj(network_folder + tax_id_2 + "_networks_string.v11.0.pckl")
    if leaveout_2:
        A_1 = sparse.identity(A_1.shape[0], dtype=float)
    if leaveout_2:
        A_2 = sparse.identity(A_2.shape[0], dtype=float)
    try:
        R = load_blastp(blast_folder + tax_id_1 + "-" + tax_id_2 + "_blastp.tab", prot2index_1, prot2index_2)
    except FileNotFoundError:
        print('Blast file for ' + str(tax_id_1) + "-" + tax_id_2 + ' not found. Blasting them and then computing block matrix with isorank.')
        interspecies_blast([tax_id_1, tax_id_2])
        R = load_blastp(blast_folder + tax_id_1 + "-" + tax_id_2 + "_blastp.tab", prot2index_1, prot2index_2)
    print('Computing isorank for ' + block_mat_fname)
    S = IsoRank(A_1, A_2, R, alpha=alpha, maxiter=1000, rand_init=rand_init, ones_init=ones_init)
    print('Dumping to ' + block_mat_fname)
    pickle.dump(S, open(block_mat_fname, "wb"), protocol=4)


def load_single_isorank_block(tax_id_combo, alpha, block_matrix_folder):
    tax_id_1 = tax_id_combo[0]
    tax_id_2 = tax_id_combo[1]
    try: 
        block_mat_fname = block_matrix_folder + tax_id_1 + "-" + tax_id_2 + "_alpha_" + str(alpha) + "_block_matrix.pckl"
        block_mat = pickle.load(open(block_mat_fname, "rb"))
    except (OSError, IOError) as e:
        block_mat_fname = block_matrix_folder + tax_id_2 + "-" + tax_id_1 + "_alpha_" + str(alpha) + "_block_matrix.pckl"
        block_mat = pickle.load(open(block_mat_fname, "rb"))
    return block_mat


def get_s_transpose_s(x):
    print(x.shape)
    return x.transpose() @ x


def save_left_out_matrix(alpha, tax_ids, left_out_tax_id, network_folder='./network_files/', block_matrix_folder='./block_matrix_files'):
    '''
    Function assumes all necessary block matrices have already been computed, and network files (including left out one, for protein ids only) have been downloaded from STRING
    '''
    tax_id_combos = []
    for ii in range(0, len(tax_ids)):
        if tax_ids[ii] != left_out_tax_id: # check to see if the tax id is the same; you don't want to have extra combos of -leavout -leaveout
            tax_id_combos.append((tax_ids[ii], left_out_tax_id + '-leaveout'))
    print(tax_id_combos)
    pool = Pool(int(multiprocessing.cpu_count()))
    isorank_blocks = pool.starmap(load_single_isorank_block, zip(tax_id_combos, itertools.repeat(alpha), itertools.repeat(block_matrix_folder)))
    print(len(isorank_blocks))
    print(isorank_blocks[0].shape)
    print(isorank_blocks)
    replacements = [get_s_transpose_s(isorank_block.todense()) for isorank_block in isorank_blocks]
    #replacements = pool.starmap(get_ss_transpose, zip(isorank_blocks))
    left_out_matrix = np.mean(replacements, axis=0)
    print(left_out_matrix.shape)
    tax_ids.remove(left_out_tax_id)
    left_out_fname = network_folder + left_out_tax_id + "_leftout_features_using_" + ','.join(tax_ids) + "_string.v11.0.pckl"
    network_file = network_folder + left_out_tax_id + "_networks_string.v11.0.pckl"
    prot2index, A, net_prots = load_adj(network_file)
    left_out_feats = {}
    left_out_feats['net'] = sparse.csr_matrix(left_out_matrix)
    left_out_feats['prot_IDs'] = net_prots
    print(left_out_feats.keys())
    print('Dumping ' + left_out_fname)
    pickle.dump(left_out_feats, open(left_out_fname, 'wb'))


def save_rwr_matrices(tax_ids, network_folder='./network_files/', leave_species_out=None, block_matrix_folder=None):
    if leave_species_out is not None:
        print('First getting sum of block matrices to get the replacement of rwr matrix for the left-out species.')
        save_left_out_matrix(tax_ids, leave_species_out, network_folder=network_folder, block_matrix_folder=block_matrix_folder)

    print('network_folder: ' + str(network_folder))
    num_cores = multiprocessing.cpu_count()
    print('num cores:')
    print(num_cores)
    pool = Pool(num_cores)
    #[pool.apply(get_single_rwr, args=(taxon, network_folder)) for taxon in tax_ids]
    pool.starmap(get_single_rwr, zip(tax_ids, itertools.repeat(network_folder)))

    '''
    for ii in range(0, len(tax_ids)):
        net = {}
        network_file = network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl"
        try:
            prot2index, A, net_prots = load_adj(network_file)
        except FileNotFoundError:
            print('Network file ' + network_file + ' not found. Downloading it and then computing rwr.')
            save_networks([tax_ids[ii]], network_folder=network_folder) # save network that is not found
            prot2index, A, net_prots = load_adj(network_file)
        net['net'] = RWR(A, maxiter=1000)
        net['prot_IDs'] = net_prots
        pickle.dump(net, open(network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "wb"))
        print ('\n')
    '''


def save_block_matrices(alpha, tax_ids, network_folder='./network_files/', blast_folder='./blast_files/', block_matrix_folder='./block_matrix_files/', rand_init=False, ones_init=False, leave_species_out=None):
    '''
    for ii in range(0, len(tax_ids)):
        prot2index_1, A_1, _ = load_adj(network_folder + tax_ids[ii] + "_networks_string.v10.5.pckl")
        for jj in range(ii+1, len(tax_ids)):
            prot2index_2, A_2, _ = load_adj(network_folder + tax_ids[jj] + "_networks_string.v10.5.pckl")
            try:
                R = load_blastp(blast_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_blastp.tab", prot2index_1, prot2index_2)
            except FileNotFoundError:
                print('Blast file for ' + str(tax_ids[ii]) + "-" + tax_ids[jj] + ' not found. Blasting them and then computing block matrix with isorank.')
                interspecies_blast([tax_ids[ii], tax_ids[jj]])
                R = load_blastp(blast_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_blastp.tab", prot2index_1, prot2index_2)
            S = IsoRank(A_1, A_2, R, alpha=alpha, maxiter=1000, rand_init=rand_init, ones_init=ones_init)
            print('Dumping to ' + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl")
            pickle.dump(S, open(block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "wb"))
    '''
    tax_id_combos = []
    if leave_species_out is not None:
        tax_ids.append(leave_species_out + '-leaveout') # add a special leaveout block matrix list to generate, in addition to regular isorank files
    for ii in range(0, len(tax_ids)):
        for jj in range(ii+1, len(tax_ids)):
            tax_id_combos.append((tax_ids[ii], tax_ids[jj]))
    pool = Pool(int(multiprocessing.cpu_count()))
    print('total combos: ' + str(len(tax_id_combos)))
    pool.starmap(get_single_isorank_block, zip(tax_id_combos, itertools.repeat(alpha), itertools.repeat(network_folder), itertools.repeat(blast_folder), itertools.repeat(block_matrix_folder), itertools.repeat(rand_init), itertools.repeat(ones_init)))
    #[pool.apply(get_single_isorank_block, args=(tax_id_combo, alpha, network_folder, blast_folder, block_matrix_folder, rand_init, ones_init)) for tax_id_combo in tax_id_combos]
    if leave_species_out is not None:
        tax_ids.remove(leave_species_out + '-leaveout')


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

