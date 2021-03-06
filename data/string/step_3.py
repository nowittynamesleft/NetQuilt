import sys
from create_block_matrix import save_block_matrices

tax_ids = sys.argv[1].split(',')
alpha = float(sys.argv[2])
if len(sys.argv) < 4:
    string_version = '11'
    print('Using default string version 11')
else:
    string_version = sys.argv[3]

net_dir = './network_files_no_add/'
save_block_matrices(alpha, tax_ids, network_folder=net_dir, blast_folder='./blast_files/', block_matrix_folder='./block_matrix_ones_init_test_files_no_add/', rand_init=False, ones_init=True)
print('Done')
