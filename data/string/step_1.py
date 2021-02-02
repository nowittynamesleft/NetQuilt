import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas
from get_annotations import save_annots

# saves networks, annotations and fastas for the taxa input as the first argument (comma delimited)

tax_ids = sys.argv[1].split(',')

net_dir = './network_files_no_add/'
save_annots(tax_ids)
fasta_fnames = get_fastas(tax_ids, fasta_folder='./fasta_files/')
save_networks(tax_ids, network_folder=net_dir)
print('Done')
