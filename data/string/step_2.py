import sys
from get_fastas_and_blasts import interspecies_blast
from create_block_matrix import save_rwr_matrices

tax_ids = sys.argv[1].split(',')

net_dir = './network_files_no_add/'

interspecies_blast(tax_ids)
save_rwr_matrices(tax_ids, network_folder=net_dir)

print('Done.')
