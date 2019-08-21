import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas, interspecies_blast
from get_annotations import save_annots
from create_block_matrix import save_block_matrices, save_rwr_matrices

tax_ids = sys.argv[1].split(',')
alpha = float(sys.argv[2])

# annotations
save_annots(tax_ids)
#save_networks(tax_ids)
#fasta_fnames = get_fastas(tax_ids)
#interspecies_blast(tax_ids)
save_rwr_matrices(tax_ids)
save_block_matrices(alpha, tax_ids)
print('Done.')
