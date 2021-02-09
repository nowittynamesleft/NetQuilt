import sys
from get_fastas_and_blasts import interspecies_blast
#from create_block_matrix import save_rwr_matrices

# blasts the downloaded fasta files for the given taxa given by the first argument (comma delimited)

tax_ids = sys.argv[1].split(',')
string_version = sys.argv[2]

net_dir = './network_files_no_add_string_v10/'

interspecies_blast(tax_ids, fasta_folder='./fasta_files_string_v10/', blast_folder='./blast_files_string_v10/', version=string_version)

print('Done.')
