from keras.models import Model, load_model
from sklearn.preprocessing import minmax_scale
from deepNF import build_AE, build_MDA
from keras.callbacks import EarlyStopping
import preprocessing as pp
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def goid2name(fname):
    goterms = []
    f = open(fname, 'r')
    for line in f:
        goid, goname = line.strip().split('\t')
        goterms.append(goname)
    f.close()

    return np.array(goterms)


def load_net(fnet, fgenes, name):
    names = ['neighborhood', 'fusion', 'cooccurence', 'coexpression',
             'experimental', 'database']
    idx = names.index(name) + 2

    genes = []
    fRead = open(fgenes, 'rb')
    for line in fRead:
        genes.append(line.strip())
    fRead.close()
    genes = np.array(genes)

    G = nx.Graph()
    for g in genes:
        G.add_node(g, color=0)
    fRead = open(fnet, 'rb')
    fRead.readline()
    fRead.readline()
    for line in fRead:
        splitted = line.strip().split()
        p1 = genes[int(splitted[0])]
        p2 = genes[int(splitted[1])]
        if int(splitted[idx]) > 0:
            G.add_edge(p1, p2)
    fRead.close()

    return G, genes


tax = '559292'
results_path = "../viz_results/"
features = {}
input_dims = {}

# Load annotations
Annot = pp.load_th_annot('../data/annot/171122-' + tax + '_th_annot.mat')
onts = ['MF', 'BP', 'CC']
goterms = goid2name('../results/' + tax + '_MF_GOterms.tsv')

print "### Load networks features [PPMI]..."
Net_features = pp.load_net_features('../data/string/' + tax + '_string_networks_steps3_ppmi.pckl')
for key in Net_features:
    features[key] = minmax_scale(Net_features[key])
    input_dims[key] = features[key].shape[1]

# parameters for AE
names = ['experimental', 'coexpression']
go_ids = [3, 9, 15]


# Training AE/MDA
X = {}
for name in names:
    print "### Load %s..." % (name)
    # Load/generate AE model
    model_name = results_path + tax + '_AE_' + name + '.h5'
    if os.path.exists(model_name):
        mid_model = load_model(model_name)
    else:
        model = build_AE(input_dims[name], [1000, 600, 1000])
        model.fit(features[name], features[name], epochs=80,
                  batch_size=64, shuffle=True, validation_split=0.2,
                  callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2)])
        # Extract encoding part of AE
        mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)
        mid_model.save(model_name)
    # protein features
    X[name] = mid_model.predict(features[name])
    X[name] = minmax_scale(X[name])

# Load/generate MDA model
print "### Load %s..." % (names[0] + '-' + names[1])
model_name = results_path + tax + '_MDA_' + names[0] + '-' + names[1] + '.h5'
if os.path.exists(model_name):
    mid_model = load_model(model_name)
else:
    model = build_MDA([input_dims[names[0]], input_dims[names[1]]], [[1000, 1000], 600])
    model.fit([features[names[0]], features[names[1]]], [features[names[0]], features[names[1]]], epochs=80,
              batch_size=64, shuffle=True, validation_split=0.2,
              callbacks=[EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001, patience=2)])
    # Extract encoding part of AE
    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)
    mid_model.save(model_name)

# protein features
X[names[0] + '-' + names[1]] = mid_model.predict([features[names[0]], features[names[1]]])
X[names[0] + '-' + names[1]] = minmax_scale(X[names[0] + '-' + names[1]])


# Annotations
GO = Annot['MF']
y_train = np.asarray(GO['y_train'])
y_valid = np.asarray(GO['y_valid'])
y_test = np.asarray(GO['y_test'])
genes = np.array(Annot['MF']['genes'])

y = np.zeros((X[names[0]].shape[0], y_train.shape[1]))
y[GO['train_idx']] = y_train
y[GO['valid_idx']] = y_valid
y[GO['test_idx']] = y_test

# delete protein/nodes without annotations
idx = np.where(y.sum(axis=1) > 0)[0]
y = y[idx]
del_nodes = [genes[ii] for ii in np.where(y.sum(axis=1) == 0)[0]]
genes = genes[idx]

print "### Number of genes = %d" % (len(idx))

# interested in go terms
c_idx = np.zeros(y.shape[0])
ii = 0
for go_id in go_ids:
    ii += 1
    c_idx[np.where(y[:, go_id] > 0)[0]] = ii

my_cmap = ['grey', 'red', 'sienna', 'cyan', 'blue', 'magenta',
           'sandybrown', 'gold', 'green', 'pink', 'darkslateblue', 'lightsalmon']


for name in names:
    # Load network
    print "### Load %s..." % (name)
    G, genes = load_net('../data/string/' + tax + '_string_networks.txt', '../data/annot/' + tax + '_genes.txt', name)

    G.remove_nodes_from(del_nodes)
    X[name] = X[name][idx]

    # t-SNE for feature representation
    print "## t-SNE for DNN features..."
    data_dnn = TSNE(n_components=2, random_state=0, perplexity=50).fit_transform(X[name])

    # Expot low-dim embedding
    plt.figure()
    plt.title('t-SNE embedding of AE features: ' + name)
    plt.scatter(data_dnn[:, 0], data_dnn[:, 1], s=10, c=c_idx, cmap=plt.cm.get_cmap("jet", len(go_ids) + 1))
    cbar = plt.colorbar(ticks=range(len(go_ids) + 1), orientation='horizontal')
    cbar.ax.set_xticklabels(np.concatenate((np.array(['other']), goterms[go_ids])), rotation=90, fontsize=14)
    plt.savefig(results_path + tax + '_' + name + '_lowdim.png', bbox_inches='tight')

    # Export networks
    for ii in range(1, len(go_ids) + 1):
        for g in genes[np.where(c_idx == ii)[0]]:
            G.node[g]['color'] = ii
    nx.write_gml(G, results_path + tax + '_' + name + '.gml')

print "### Load %s..." % (names[0] + '-' + names[1])
X[names[0] + '-' + names[1]] = X[names[0] + '-' + names[1]][idx]

# t-SNE for feature representation
print "## t-SNE for DNN features..."
data_dnn = TSNE(n_components=2, random_state=0, perplexity=50).fit_transform(X[names[0] + '-' + names[1]])

# Expot low-dim embedding
plt.figure()
plt.title('t-SNE embedding of AE features: ' + names[0] + '-' + names[1])
plt.scatter(data_dnn[:, 0], data_dnn[:, 1], s=10, c=c_idx, cmap=plt.cm.get_cmap("jet", len(go_ids) + 1))
cbar = plt.colorbar(ticks=range(len(go_ids) + 1), orientation='horizontal')
cbar.ax.set_xticklabels(np.concatenate((np.array(['other']), goterms[go_ids])), rotation=90, fontsize=14)
plt.savefig(results_path + tax + '_' + names[0] + '-' + names[1] + '_lowdim.png', bbox_inches='tight')
