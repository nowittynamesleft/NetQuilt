import sys
from string2adj import save_networks
from get_fastas_and_blasts import get_fastas
from get_annotations import save_annots

# saves networks, annotations and fastas for the taxa input as the first argument (comma delimited)

tax_ids = sys.argv[1].split(',')
min_coverage = float(sys.argv[2]) # Minimum percentage of proteins of selected taxa a GO term must annotate to be included
max_coverage = float(sys.argv[3]) # Maximum percentage of proteins of selected taxa a GO term must annotate to be included
if len(sys.argv) < 5:
    string_version = '11'
    print('Using default string version 11')
else:
    string_version = sys.argv[4]

net_dir = './network_files_no_add/'
fasta_fnames = get_fastas(tax_ids, fasta_folder='./fasta_files/', version=string_version)
save_annots(tax_ids, min_coverage=min_coverage, max_coverage=max_coverage, version=string_version)
save_networks(tax_ids, network_folder=net_dir, version=string_version)
print('Done')
