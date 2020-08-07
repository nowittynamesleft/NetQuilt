import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas, interspecies_blast
from get_annotations import save_annots
from create_block_matrix import save_block_matrices, save_rwr_matrices, save_left_out_matrix, test_leaveout_calculations
import argparse
import os


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print('Creating directory ' + directory)
        os.makedirs(directory) 
        

parser = argparse.ArgumentParser(description='Pipeline for preprocessing data for Multispecies Maxout.')
parser.add_argument('--tax_ids', type=str, help='Tax ids delimited by \',\' to generate isorank matrices for')
parser.add_argument('--alpha', type=float, help='alpha value for isorank')
parser.add_argument('--leave_species_out', type=str, default=None, help='species to leave out')
parser.add_argument('--left_out_mat_version', type=int, default=None, help='Version of projection (1, 2, 3, 4) used for left-out matrix')
parser.add_argument('--set_iterations', type=int, default=None, help='Number of iterations to manually set IsoRank to run')
args = parser.parse_args()

tax_ids = args.tax_ids.split(',')
print(tax_ids)
alpha = args.alpha
leave_species_out = args.leave_species_out
leftout_mat_version = args.left_out_mat_version
set_iterations = args.set_iterations

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
#block_mat_folder = './block_matrix_ones_init_test_files_no_add/'
block_mat_folder = './block_matrix_ones_init_no_add_set_iter_' + str(set_iterations) + '/'
ensure_dir(block_mat_folder)
blast_folder = './blast_files/'
#blast_folder = './blast_test_folder/'
#block_mat_folder = './block_matrix_test_folder/'

#print('Saving all leave out matrices for ' + leave_species_out)
save_block_matrices(alpha, tax_ids, network_folder=net_dir, blast_folder=blast_folder, block_matrix_folder=block_mat_folder, rand_init=False, ones_init=True, leave_species_out=leave_species_out, set_iterations=set_iterations)
#save_left_out_matrix(alpha, tax_ids, leave_species_out, blast_folder=blast_folder, network_folder=net_dir, block_matrix_folder=block_mat_folder, version=leftout_mat_version)
#print('Testing leaveout calcs for ' + leave_species_out)
#test_leaveout_calculations(alpha, tax_ids, leave_species_out, network_folder=net_dir, blast_folder=blast_folder, block_matrix_folder=block_mat_folder)
print('Done.')
