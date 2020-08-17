from isorank import RWR, IsoRank, isorank_leaveout_test
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
from sklearn.preprocessing import minmax_scale
import os


# Edited for STRING v11 instead of v10.5
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.makedirs(directory) 


def load_adj(fname_pckl, net_type='experimental'):
    """Load pickle file with string networks."""
    file_pckl = open(fname_pckl, 'rb')
    net = pickle.load(file_pckl)
    file_pckl.close()
    proteins = net['prot_IDs']
    if net_type is not 'projected':
        A = net['nets'][net_type]
    else:
        A = net['net']
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

def load_blast_from_taxa(tax_id_1, tax_id_2, prot2index_1, prot2index_2, blast_folder):
    try:
        R = load_blastp(blast_folder + tax_id_1 + "-" + tax_id_2 + "_blastp.tab", prot2index_1, prot2index_2)
    except FileNotFoundError:
        print('Blast file for ' + tax_id_1 + "-" + tax_id_2 + ' not found. Switching the tax ids to see if the transpose can be found..')
        try: 
            blast_file = blast_folder + tax_id_2 + "-" + tax_id_1 + "_blastp.tab" # loads blast file in a way that produces same shape as what was originally tried in the try clause above
            R = load_blastp(blast_file, prot2index_1, prot2index_2)
            print('Blast file ' + blast_file + ' found!')
            print('R shape (not transposed):')
            print(R.shape)
        except FileNotFoundError:
            print('Blast file for ' + tax_id_2 + "-" + tax_id_1 + ' not found. Blasting them and then computing block matrix with isorank.')
            interspecies_blast([tax_id_1, tax_id_2])
            R = load_blastp(blast_folder + tax_id_1 + "-" + tax_id_2 + "_blastp.tab", prot2index_1, prot2index_2)
    return R



def get_single_rwr(tax_id, network_folder):
    net = {}
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
    pickle.dump(net, open(rwr_fname, "wb"), protocol=4)
    print('Saved ' + rwr_fname)
    print ('\n')


def save_single_isorank_block(tax_id_combo, alpha, network_folder, blast_folder, block_matrix_folder, rand_init, ones_init, used_tax_ids=None, version=None, set_iterations=None):
    print('Set iterations inside save_single_isorank_block: ' + str(set_iterations))
    tax_id_1 = tax_id_combo[0]
    tax_id_2 = tax_id_combo[1]
    if used_tax_ids is None: # for left out matrix IsoRank calculation
        block_mat_fname = block_matrix_folder + tax_id_1 + "-" + tax_id_2 + "_alpha_" + str(alpha) + "_block_matrix.pckl"
    else:
        block_mat_fname = block_matrix_folder + tax_id_1 + "-" + tax_id_2 + "_left_out_using_" + ",".join(used_tax_ids) + '_version_' + str(version) + "_alpha_" + str(alpha) + "_block_matrix.pckl"
    leaveout_1 = False
    leaveout_2 = False
    try:
        assert not ('-leaveout' in tax_id_1 and '-leaveout' in tax_id_2)
    except AssertionError:
        print('Both matrices are left out. Not creating IsoRank for this pair.')
        return
    if '-leaveout' in tax_id_1:
        tax_id_1 = tax_id_1.split('-')[0] # get actual tax id to load network file for the protein ids, not the adjacency matrix, and also for the blast file
        leaveout_1 = True
    elif '-leaveout' in tax_id_2:
        tax_id_2 = tax_id_2.split('-')[0]
        leaveout_2 = True

    if path.exists(block_mat_fname):
        print('Block mat file ' + block_mat_fname + ' already exists; skipping')
        return
    '''
    elif tax_id_1 == tax_id_2:
        print('No isorank matrix will be computed between a species with itself')
        return
    '''
    prot2index_1, A_1, _ = load_adj(network_folder + tax_id_1 + "_networks_string.v11.0.pckl")
    prot2index_2, A_2, _ = load_adj(network_folder + tax_id_2 + "_networks_string.v11.0.pckl")
    if leaveout_1:
        A_1 = sparse.identity(A_1.shape[0], dtype=float)
    elif leaveout_2:
        A_2 = sparse.identity(A_2.shape[0], dtype=float)
    if used_tax_ids is not None: # the case where we want isorank for a predicted left out network with itself
        try:
            assert tax_id_1 == tax_id_2
        except AssertionError:
            print('Used tax ids supplied to save_single_isorank_block but tax_id_1 and tax_id_2 are not the same.')
            print('Tax id 1: ' + tax_id_1)
            print('Tax id 2: ' + tax_id_2)
        left_out_fname = network_folder + tax_id_1 + "_leftout_network_using_" + ','.join(used_tax_ids) + "_string.v11.0.pckl" 
        print('Loading left out network ' + left_out_fname)
        prot2index_1, A_1, _ = load_adj(left_out_fname, net_type='projected')
        prot2index_2, A_2, _ = load_adj(left_out_fname, net_type='projected')
        
    R = load_blast_from_taxa(tax_id_1, tax_id_2, prot2index_1, prot2index_2, blast_folder)
    print('Computing isorank for ' + block_mat_fname)
    print('A_1 shape:')
    print(A_1.shape)
    print('A_2 shape:')
    print(A_2.shape)
    S = IsoRank(A_1, A_2, R, alpha=alpha, maxiter=50, rand_init=rand_init, ones_init=ones_init, set_iterations=set_iterations)
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


def get_s_transpose_A_s(S_12, A_1):
    A_hat_2 = S_12.transpose()@ A_1 @ S_12
    return A_hat_2


def save_left_out_matrix(alpha, tax_ids, left_out_tax_id, blast_folder='./blast_files/', network_folder='./network_files/', block_matrix_folder='./block_matrix_files', version=1):
    '''
    Function assumes all necessary block matrices have already been computed, and network files (including left out one, for protein ids only) have been downloaded from STRING
    Need to make a function to compute S^{T}S (bipartite graph projection) for every IsoRank matrix related to the left-out matrix,
    and then averages them to get the predicted network. Save this network, and then using BLAST matrix of species with itself and compute IsoRank between the predicted
    network and the BLAST network.
    '''
    print('Save left out matrix!')
    print('Tax ids:')
    print(tax_ids)
    tax_id_combos = []
    used_tax_ids = [tax_id for tax_id in tax_ids if tax_id != left_out_tax_id]
    for ii in range(0, len(tax_ids)):
        tax_id_combos.append((tax_ids[ii], left_out_tax_id + '-leaveout'))
    print(tax_id_combos)
    pool = Pool(int(multiprocessing.cpu_count()))
    #isorank_blocks = pool.starmap(load_single_isorank_block, zip(tax_id_combos, itertools.repeat(alpha), itertools.repeat(block_matrix_folder)))
    network_file = network_folder + left_out_tax_id + "_networks_string.v11.0.pckl"
    leftout_prot2index, A, left_out_net_prots = load_adj(network_file)
    if version == 1: # S transpose S
        print('VERSION 1 (S^{T}S)')
        isorank_blocks = [load_single_isorank_block(*args) for args in zip(tax_id_combos, itertools.repeat(alpha), itertools.repeat(block_matrix_folder))]
        replacements = [get_s_transpose_s(isorank_block.todense()) for isorank_block in isorank_blocks]
    elif version == 2: # S matrix network projection
        print('VERSION 2 (S^{T}AS) S MATRIX NETWORK PROJECTION WITH NONLEFTOUT ORGANISM\'S NETWORK')
        replacements = []
        for tax_id_combo in tax_id_combos:
            nonleftout_taxon = tax_id_combo[0]
            network_file = network_folder + nonleftout_taxon + "_networks_string.v11.0.pckl"
            _, nonleftout_net, _ = load_adj(network_file)
            isorank_block = load_single_isorank_block(tax_id_combo, alpha, block_matrix_folder)
            replacements.append(get_s_transpose_A_s(isorank_block.todense(), nonleftout_net.todense()))
    elif version == 3: # blast only baseline
        print('VERSION 3 (R^{T}R) BLAST ONLY')
        replacements = []
        for tax_id_combo in tax_id_combos:
            nonleftout_taxon = tax_id_combo[0]
            network_file = network_folder + nonleftout_taxon + "_networks_string.v11.0.pckl"
            prot2index_1, _, _ = load_adj(network_file)
            R = load_blast_from_taxa(nonleftout_taxon, left_out_tax_id, prot2index_1, leftout_prot2index, blast_folder)
            replacements.append(get_s_transpose_s(R.todense()))
    elif version == 4: # blast network projection
        print('VERSION 4 (R^{T}AR) BLAST NETWORK PROJECTION')
        replacements = []
        for tax_id_combo in tax_id_combos:
            nonleftout_taxon = tax_id_combo[0]
            network_file = network_folder + nonleftout_taxon + "_networks_string.v11.0.pckl"
            prot2index_1, nonleftout_net, _ = load_adj(network_file)
            nonleftout_net = nonleftout_net.todense()
            R = load_blast_from_taxa(nonleftout_taxon, left_out_tax_id, prot2index_1, leftout_prot2index, blast_folder).todense()
            replacements.append(get_s_transpose_A_s(R, nonleftout_net))
    else:
        raise NotImplementedError('Version for making left out network matrix must be either 1, 2, 3, 4.')
    replacements = np.array(replacements)
    print(replacements.shape)

    #replacements = pool.starmap(get_ss_transpose, zip(isorank_blocks))
    left_out_matrix = np.mean(replacements, axis=0)
    print(left_out_matrix.shape)
    density = np.count_nonzero(left_out_matrix)/(left_out_matrix.shape[0]*left_out_matrix.shape[1])
    print(left_out_matrix)
    print('Density of left out matrix: ' + str(density))
    left_out_matrix = minmax_scale(left_out_matrix)
    print(left_out_matrix)
    density = np.count_nonzero(left_out_matrix)/(left_out_matrix.shape[0]*left_out_matrix.shape[1])
    print('Density of left out matrix after minmax scaling: ' + str(density))
    left_out_fname = network_folder + left_out_tax_id + "_leftout_network_using_" + ','.join(used_tax_ids) + '_version_' + str(version) + "_string.v11.0.pckl"
    left_out_feats = {}
    left_out_feats['net'] = sparse.csr_matrix(left_out_matrix)
    left_out_feats['prot_IDs'] = left_out_net_prots
    print(left_out_feats.keys())
    print('Dumping ' + left_out_fname)
    pickle.dump(left_out_feats, open(left_out_fname, 'wb'), protocol=4)
    print('Making IsoRank block of leaveout species with intraspecies blast connections')
    save_single_isorank_block((left_out_tax_id, left_out_tax_id), alpha, network_folder, blast_folder, block_matrix_folder, False, True, used_tax_ids=used_tax_ids, version=version)


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


def save_block_matrices(alpha, tax_ids, network_folder='./network_files/', blast_folder='./blast_files/', block_matrix_folder='./block_matrix_files/', rand_init=False, ones_init=False, leave_species_out=None, set_iterations=None):
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
    ensure_dir(block_matrix_folder)
    tax_id_combos = []
    taxa = list(tax_ids)
    if leave_species_out is not None:
        taxa.append(leave_species_out + '-leaveout') # add combos to isorank using identity matrix
    '''
    for ii in range(0, len(tax_ids)):
        for jj in range(ii+1, len(tax_ids)):
            tax_id_combos.append((tax_ids[ii], tax_ids[jj]))
    '''
    for ii in range(0, len(taxa)):
        for jj in range(ii, len(taxa)): # now including species with themselves
            if not ('-leaveout' in taxa[ii] and '-leaveout' in taxa[jj]): # but do not make leaveout in both
                tax_id_combos.append((taxa[ii], taxa[jj]))


    pool = Pool(int(multiprocessing.cpu_count()))
    print('total combos: ' + str(len(tax_id_combos)))
    print('Set iterations: ' + str(set_iterations))
    pool.starmap(save_single_isorank_block, zip(tax_id_combos, itertools.repeat(alpha), itertools.repeat(network_folder), 
        itertools.repeat(blast_folder), itertools.repeat(block_matrix_folder), itertools.repeat(rand_init), 
        itertools.repeat(ones_init), itertools.repeat(None), itertools.repeat(None), 
        itertools.repeat(set_iterations))) # this is not the call for leave out species; this is only for non-leaveout. Use "save_left_out_matrix" instead


def test_leaveout_calculations(alpha, tax_ids, leave_species_out, network_folder='./network_files/', blast_folder='./blast_files/', block_matrix_folder='./block_matrix_files/'):
    tax_id_combos = []
    taxa = list(tax_ids)
    for ii in range(0, len(taxa)):
        if taxa[ii] != leave_species_out:
            tax_id_combos.append((taxa[ii], leave_species_out + '-leaveout'))
    print('total combos: ' + str(len(tax_id_combos)))
    for tax_id_combo in tax_id_combos:
        tax_id_1 = tax_id_combo[0]
        tax_id_2 = tax_id_combo[1]
        block_mat_fname = block_matrix_folder + tax_id_1 + "-" + tax_id_2 + "_alpha_" + str(alpha) + "_block_matrix.pckl"
        leaveout_1 = False
        leaveout_2 = False
        try:
            assert not ('-leaveout' in tax_id_1 and '-leaveout' in tax_id_2)
        except AssertionError:
            print('Both matrices are left out. Not creating IsoRank for this pair.')
            return
        if '-leaveout' in tax_id_1:
            tax_id_1 = tax_id_1.split('-')[0] # get actual tax id to load network file for the protein ids, not the adjacency matrix, and also for the blast file
            leaveout_1 = True
        elif '-leaveout' in tax_id_2:
            tax_id_2 = tax_id_2.split('-')[0]
            leaveout_2 = True

        if path.exists(block_mat_fname):
            print('Block mat file ' + block_mat_fname + ' already exists; skipping')
            return
        '''
        elif tax_id_1 == tax_id_2:
            print('No isorank matrix will be computed between a species with itself')
            return
        '''
        prot2index_1, A_1, _ = load_adj(network_folder + tax_id_1 + "_networks_string.v11.0.pckl")
        prot2index_2, A_2, _ = load_adj(network_folder + tax_id_2 + "_networks_string.v11.0.pckl")
        if leaveout_1:
            A_1 = sparse.identity(A_1.shape[0], dtype=float)
        elif leaveout_2:
            A_2 = sparse.identity(A_2.shape[0], dtype=float)
            
        R = load_blast_from_taxa(tax_id_1, tax_id_2, prot2index_1, prot2index_2, blast_folder)
        print('Computing isorank for ' + block_mat_fname)
        print('A_1 shape:')
        print(A_1.shape)
        print('A_2 shape:')
        print(A_2.shape)

    isorank_leaveout_test(A_1, R, alpha)



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

