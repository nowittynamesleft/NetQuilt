import sys
from create_block_matrix import save_block_matrices

tax_ids = sys.argv[1].split(',')
alpha = float(sys.argv[2])
string_version = sys.argv[3]

net_dir = './network_files_no_add_string_v10/'
save_block_matrices(alpha, tax_ids, network_folder=net_dir, blast_folder='./blast_files_string_v10/', block_matrix_folder='./block_matrix_ones_init_test_files_no_add_string_v10/', rand_init=False, ones_init=True)
print('Done')
