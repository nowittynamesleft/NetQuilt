import pickle
import sys
from Bio import SeqIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

annots = pickle.load(open(sys.argv[1], 'rb'))
annotated_taxa = annots.keys()
prot_id_fname = sys.argv[2]
taxa_domains_df = pd.read_csv(sys.argv[3], sep='\t', index_col=False, dtype={'tax_id': str, 'domain': str, 'name_txt': str})
taxa_to_names = dict(zip(taxa_domains_df.tax_id, taxa_domains_df.name_txt))

domain_full_name = {'E': 'Eukaryota',
                    'B': 'Bacteria',
                    'A': 'Archea'
}

for domain in ['E', 'B', 'A']:
    prot_id_file = open(prot_id_fname, 'r')
    print('Making figure for ' + domain_full_name[domain])
    domain_taxa = set(taxa_domains_df[taxa_domains_df.domain == domain]['tax_id'])
    total_taxa_prots = {}
    for line in prot_id_file:
        tax_id = str(line.split('.')[0])
        if tax_id in domain_taxa:
            if tax_id not in total_taxa_prots:
                total_taxa_prots[tax_id] = 0
            else:
                total_taxa_prots[tax_id] += 1
    prot_id_file.close()

    percentages = []
    for taxon in annotated_taxa:
        if taxon in domain_taxa:
            num_annotated_prots = np.sum(annots[taxon]['annot']['molecular_function'].todense().any(axis=1) + annots[taxon]['annot']['cellular_component'].todense().any(axis=1) + annots[taxon]['annot']['biological_process'].todense().any(axis=1))
            percentages.append((taxon, float(num_annotated_prots)/total_taxa_prots[taxon]))
    sorted_percentages = sorted(percentages, key=lambda x: x[1], reverse=True)
    k = 30
    top_k_tuples = sorted_percentages[:k]
    taxa = [tup[0] for tup in top_k_tuples]
    percs = [tup[1]*100 for tup in top_k_tuples]

# # Plot # #
    ind = np.arange(k)  # the x locations for the groups
    width = 0.5       # the width of the bars
    N = max(ind)

    cs = ['skyblue', 'purple']

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    percent_threshold = 20
    print(ind)
    print(percs)
    red_percs = [perc for perc in percs if perc > percent_threshold]
    red_ind = ind[:len(red_percs)]
    blue_percs = [perc for perc in percs if perc <= percent_threshold]
    blue_ind = ind[len(red_percs):]
    ax.bar(red_ind, red_percs, width, color='red', align='center')
    ax.bar(blue_ind, blue_percs, width, color='blue', align='center')
    ax.set_ylabel("Percentage of Proteins Annotated", fontsize=18)
    ax.set_xlabel("Taxa", fontsize=18)
    ax.set_xticks(ind)
    ax.set_title('Top ' + str(k) + ' Species in ' + domain_full_name[domain] + ' by Percentage of Proteome with Annotations in Any GO Branch', fontsize=20)
    names = [taxa_to_names[taxon] for taxon in taxa]    
    counts = [total_taxa_prots[taxon] for taxon in taxa]
    ax.set_xticklabels(names, fontsize=12, rotation=30, rotation_mode="anchor")
    ax.set_ylim(0, 1.2*max(percs))
    for tick in ax.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    for i, v in enumerate(percs):
            ax.text(i, v + 5*max(percs)/85, str(int(counts[i]*v/100)), color='black', ha='center', fontsize=8)
            ax.text(i, v + 3*max(percs)/85, 'of', color='black', ha='center', fontsize=8)
            ax.text(i, v + max(percs)/85, str(counts[i]), color='black', ha='center', fontsize=8)
    print('Total proteins in selected organisms:')
    print(np.sum(np.array(counts)[red_ind]))
    print(','.join(np.array(taxa)[red_ind]))
    plt.savefig(domain_full_name[domain] + 'annot_stats.png', bbox_inches='tight')
# plt.show()
