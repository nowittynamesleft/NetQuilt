import sys
from get_fastas_and_blasts import interspecies_blast
#from create_block_matrix import save_rwr_matrices

# blasts the downloaded fasta files for the given taxa given by the first argument (comma delimited)

tax_ids = sys.argv[1].split(',')
string_version = sys.argv[2]

net_dir = './network_files_no_add/'

interspecies_blast(tax_ids, fasta_folder='./fasta_files/', blast_folder='./blast_files/', version=string_version)

print('Done.')
