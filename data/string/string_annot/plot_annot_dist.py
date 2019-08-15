"""
========
Barchart (final results)
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import pickle
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


single_pckl = str(sys.argv[1])
go_type = str(sys.argv[2])


single_annot = pickle.load(open(single_pckl, 'rb'))
single_Y = single_annot['annot'][go_type].toarray()
single_goids = single_annot['go_IDs'][go_type]
single_gonames = single_annot['go_names'][go_type]
single_prots = np.asarray(single_annot['prot_IDs'])

test_funcs = np.where(np.logical_and(single_Y.sum(axis=0) > 30, single_Y.sum(axis=0) <= 150))[0]

common_goids = []
common_gonames = []
for ii in test_funcs:
    go_id = single_goids[ii]
    go_name = single_gonames[ii]
    indx = np.where(single_Y[:, ii] > 0)[0]
    tmp_prots = single_prots[indx]
    tax = set([prot.split('.')[0] for prot in tmp_prots])
    if len(tax) == 6:
        common_goids.append(go_id)
        common_gonames.append(go_name)
        print (go_id, '\t', go_name)


pickle.dump(common_goids, open("bacteria_" + go_type + "_train_goids.pckl", "wb"))


"""
multi_pckl = str(sys.argv[2])

single_annot = pickle.load(open(single_pckl, 'rb'))
multi_annot = pickle.load(open(multi_pckl, 'rb'))

single_Y = single_annot['annot'][go_type].toarray()
single_goids = single_annot['go_IDs'][go_type]
single_gonames = single_annot['go_names'][go_type]

multi_Y = multi_annot['annot'][go_type].toarray()
multi_prots = multi_annot['prot_IDs']
idx = [ii for ii, prot in enumerate(multi_prots) if not prot.startswith("553174")]
multi_prots = np.asarray(multi_prots)
multi_prots = multi_prots[idx]
multi_Y = multi_Y[idx]
multi_goids = multi_annot['go_IDs'][go_type]

test_funcs = np.where(np.logical_and(single_Y.sum(axis=0) > 30, single_Y.sum(axis=0) <= 100))[0]

common_goids = []
common_gonames = []
for ii in test_funcs:
    go_id = single_goids[ii]
    go_name = single_gonames[ii]
    if go_id in multi_goids:
        idx = multi_goids.index(go_id)
        if sum(multi_Y[:, idx]) > 50:
            indx = np.where(multi_Y[:, idx] > 0)[0]
            tmp_prots = multi_prots[indx]
            tax = set([prot.split('.')[0] for prot in tmp_prots])
            if len(tax) == 5:
                common_goids.append(go_id)
                common_gonames.append(go_name)
                print (go_id, '\t', go_name)


pickle.dump(common_goids, open("553174-model-org_" + go_type + "_train_goids.pckl", "wb"))

single_idx = [single_goids.index(go) for go in common_goids]
multi_idx = [multi_goids.index(go) for go in common_goids]

single_Y = single_Y[:, single_idx]
multi_Y = multi_Y[:, multi_idx]

single_count = single_Y.sum(axis=0)
multi_count = multi_Y.sum(axis=0)
multi_count += single_count

# # Plot # #
ind = np.arange(len(common_goids))  # the x locations for the groups
width = 0.20       # the width of the bars
N = max(ind)


cs = ['skyblue', 'purple']

fig, ax = plt.subplots(1, 1)

ax.bar(ind, single_count, width, color=cs[0], align='center')
ax.bar(ind + width, multi_count, width, color=cs[1], align='center')
ax.set_ylabel("# Proteins", fontsize=18)
# ax.set_title('Yeast: ' + yeast_annot2name[yeast_level], fontsize=18)
# axarr[0].set_xlim([-0.1, N])
# ax.set_ylim([0, max(yeast_deepnf_res[yeast_level])+0.05])
ax.legend(('H', 'H + Y + M + F + W + B'), fontsize=18, loc='upper left')
ax.set_xticks(ind+width)

ax.set_xticklabels(common_gonames, fontsize=8, rotation=30, rotation_mode="anchor")
for tick in ax.xaxis.get_majorticklabels():
    tick.set_horizontalalignment("right")

plt.savefig('annot_stats.png', bbox_inches='tight')
# plt.show()
"""
