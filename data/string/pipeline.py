import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas, interspecies_blast
from get_annotations import save_annots
from create_block_matrix import save_block_matrices, save_rwr_matrices, save_left_out_matrix 

tax_ids = sys.argv[1].split(',')
alpha = float(sys.argv[2])
leave_species_out = sys.argv[3]
if leave_species_out == 'None':
    leave_species_out = None

net_dir = './network_files_no_add/'
#save_annots(tax_ids)
#save_networks(tax_ids, network_folder=net_dir)
#fasta_fnames = get_fastas(tax_ids)
#interspecies_blast(tax_ids)

#save_rwr_matrices(tax_ids, network_folder=net_dir)

'''
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_rand_init_test_files_2/', rand_init=True, ones_init=False, add=True)
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_blast_init_test_files_no_add/', rand_init=False, ones_init=False)
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_rand_init_test_files_no_add/', rand_init=True, ones_init=False)
'''
block_mat_folder = './block_matrix_ones_init_test_files_no_add/'
blast_folder = './blast_files/'
#blast_folder = './blast_test_folder/'
#block_mat_folder = './block_matrix_test_folder/'

print('Saving all leave out matrices for ' + leave_species_out)
save_block_matrices(alpha, tax_ids, network_folder=net_dir, blast_folder=blast_folder, block_matrix_folder=block_mat_folder, rand_init=False, ones_init=True, leave_species_out=leave_species_out)
save_left_out_matrix(alpha, tax_ids, leave_species_out, blast_folder=blast_folder, network_folder=net_dir, block_matrix_folder=block_mat_folder)
print('Done.')
