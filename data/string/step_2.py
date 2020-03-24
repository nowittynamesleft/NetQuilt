import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas, interspecies_blast
from get_annotations import save_annots
from create_block_matrix import save_block_matrices, save_rwr_matrices

tax_ids = sys.argv[1].split(',')

net_dir = './network_files_no_add/'

interspecies_blast(tax_ids)
save_rwr_matrices(tax_ids, network_folder=net_dir)

print('Done.')
