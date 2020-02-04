import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas, interspecies_blast
from get_annotations import save_annots
from create_block_matrix import save_block_matrices, save_rwr_matrices

tax_ids = sys.argv[1].split(',')
alpha = float(sys.argv[2])

net_dir = './network_files_no_add/'
#save_annots(tax_ids)
#save_networks(tax_ids, network_folder=net_dir)
#fasta_fnames = get_fastas(tax_ids)
#interspecies_blast(tax_ids)

save_rwr_matrices(tax_ids, network_folder=net_dir)

'''
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_rand_init_test_files_2/', rand_init=True, ones_init=False, add=True)
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_blast_init_test_files_no_add/', rand_init=False, ones_init=False)
#save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_rand_init_test_files_no_add/', rand_init=True, ones_init=False)
'''
save_block_matrices(alpha, tax_ids, block_matrix_folder='./block_matrix_ones_init_test_files_no_add/', rand_init=False, ones_init=True)
print('Done.')
