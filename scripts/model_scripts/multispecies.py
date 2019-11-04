import matplotlib as mpl
mpl.use('Agg')
import sys
import pickle
import numpy as np
import os.path
# import scipy.io as sio
from scipy import stats

from deepNF import build_MDA, build_AE, build_denoising_AE, build_denoising_MDA
from validation import cross_validation, cross_validation_nn, temporal_holdout, output_projection_files, leave_one_species_out_val
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import matplotlib.pyplot as plt
import pandas as pd


BATCH_SIZE = 128
NB_EPOCH = 100
#NB_EPOCH = 1
LR = 0.01
RESULTS_PATH = '../results/loso_testing/'
#RESULTS_PATH = '../results/alpha_testing/'

# python multispecies.py annot_fname ont model_name data_folder tax_ids alpha test_go_id_fname
# example for running autoencoder on human and testing on human on the goids chosen from model-org go ids: python multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function human_only /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl

#For running autoencoder on model orgs and testing on human (with alpha=0.6):
#python multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function model_orgs_human_test /mnt/ceph/users/vgligorijevic/PFL/data/string/ 511145,7227,10090,6239,4932,9606 0.6 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl

# For temporal holdout:
# example for running autoencoder on human and testing on human on the goids chosen from model-org go ids: python multispecies.py ../../data/temporal_holdout/long_time_test_human_MFannotation_data.pkl molecular_function human_only_temporal_holdout_impl_test /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0 

# python multispecies.py annot_fname ont model_name data_folder tax_ids alpha test_go_id_fname test_tax_id
def aupr(label, score):
    """Computing real AUPR"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)

    return pr


def macro_aupr(y_test, y_score, goterms=None, out_fname=None):
    # generate indices for bootstraps
    bootstraps = []
    for i in range(0, 1000):
        bootstraps.append(resample(np.arange(y_test.shape[0])))

    # Compute macro-averaged AUPR
    goterm_perf = {}
    for i in range(y_test.shape[1]):
        goterm_perf[i] = []
        for ind in bootstraps:
            goterm_perf[i].append(aupr(y_test[ind, i], y_score[ind, i]))

    perf = 0.0
    for goid in goterm_perf:
        perf += np.mean(goterm_perf[goid])
    perf /= len(goterm_perf)

    # output file
    if goterms is not None:
        fout = open(out_fname, 'w')
        for i in range(y_test.shape[1]):
            fout.write('%s %0.4f %0.4f\n' % (goterms[i], np.mean(goterm_perf[i]), stats.sem(goterm_perf[i])))
        fout.close()
    return perf


def export_history(history, model_name, kwrd, results_path=RESULTS_PATH):
    # Export figure: loss vs epochs (history)
    plt.figure()
    plt.plot(history.history['loss'], '-')
    plt.plot(history.history['val_loss'], '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(results_path + model_name.split('-')[0] + '_' + kwrd + '_loss.png', bbox_inches='tight')


def build_model(X, input_dims, arch, mtype='mae', nf=0.5, std=1.0, epochs=NB_EPOCH, batch_size=BATCH_SIZE):
    if mtype == 'mae':
        #model = build_MDA(input_dims, arch)
        model = build_denoising_MDA(input_dims, arch)
    elif mtype == 'ae':
        print('hidden activation tanh')
        #model = build_AE(input_dims[0], arch, hidden_activation='tanh')
        model = build_denoising_AE(input_dims[0], arch, hidden_activation='tanh')
    else:
        print ("### Wrong model.")
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)
        X_train = []
        X_test = []
        for jj in range(0, len(Xs), 2):
            X_train.append(Xs[jj])
            X_test.append(Xs[jj+1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)
        for ii in range(0, len(X_train)):
            X_train_noisy[ii] = X_train_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train[ii].shape)
            X_test_noisy[ii] = X_test_noisy[ii] + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test[ii].shape)
            X_train_noisy[ii] = np.clip(X_train_noisy[ii], 0, 1)
            X_test_noisy[ii] = np.clip(X_test_noisy[ii], 0, 1)
    else:
        X_train, X_test = train_test_split(X, test_size=0.2) ### coping once
        #X_train_noisy = X_train.copy() # copying again
        #X_test_noisy = X_test.copy()
        #X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape) # now adding noise, another multiple of the original matrix in terms of memory
        #X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        #X_train_noisy = np.clip(X_train_noisy, 0, 1)
        #X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                        validation_data=(X_test, X_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

    return model, history


def get_common_indices(annot_prots, string_prots):
    common_prots = list(set(string_prots).intersection(annot_prots))
    print ("### Number of prots in intersection:", len(common_prots))
    annot_idx = [annot_prots.index(prot) for prot in common_prots]
    string_idx = [string_prots.index(prot) for prot in common_prots]

    return annot_idx, string_idx


def build_NN(input_dim, encoding_dims, go_edgelist=None):
    """
    Funciton for building Neural Network (NN) model.
    """
    # input layer
    input_layer = Input(shape=(input_dim, ), name='input')
    hidden_layer = input_layer
    for dim in encoding_dims[:-1]:
        hidden_layer = Dense(dim, activation='sigmoid')(hidden_layer)

    # reconstruction of the input
    output_layer = Dense(encoding_dims[-1],
                         activation='sigmoid',
                         name='output')(hidden_layer)
    # NN model
    optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
    # optimizer = RMSprop(lr=0.0001)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print (model.summary())

    return model


def train_NN_model(X, Y, encoding_dims, epochs=NB_EPOCH, batch_size=BATCH_SIZE):
    if isinstance(X, list):
        X_train, X_valid = X
        Y_train, Y_valid = Y
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
    else:
        X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
        input_dim = X_train.shape[1]
        output_dim = Y_train.shape[1]
    encoding_dims.append(output_dim)

    # Build the model
    model = build_NN(input_dim, encoding_dims)
    # Fitting the model
    history = model.fit(X_train, Y_train, epochs=epochs,
                        batch_size=batch_size, shuffle=True,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1)])

    return model, history


def get_mapping_dict(mapping_fname): 
    mapping_df = pd.read_csv(mapping_fname, sep='\t', names=['UniprotKB_AC', 'ID_type', 'ID'])
    mapping_df = mapping_df[mapping_df.ID_type == 'STRING']
    mapping_dict = pd.Series(mapping_df.ID.values, index=mapping_df.UniprotKB_AC).to_dict()
    return mapping_dict


def uniprot_to_string_mapping(uniprot_prots, mapping_dict):
    string_ids = []
    missing_inds = []
    for i, prot in enumerate(uniprot_prots):
        if prot in mapping_dict:
            string_ids.append(mapping_dict[prot])
        else:
            missing_inds.append(i)
    missing_inds = np.array(missing_inds)
    return string_ids, missing_inds 


def remove_missing_annot_prots(annot_prots, Y, mapping_dict):
    new_annot_prots, missing_inds = uniprot_to_string_mapping(annot_prots, mapping_dict)
    Y = np.delete(Y, missing_inds, axis=0)
    return new_annot_prots, Y


def temporal_holdout_main(annot_fname, model_name, data_folder, tax_ids, alpha, mapping_fname):
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    # selected goids
    test_funcs = Annot['func_inds']
    train_inds = Annot['train_prot_inds']
    valid_inds = Annot['valid_prot_inds']
    test_inds = Annot['test_prot_inds']
    Y_train = np.asarray(Annot['train_annots'])[:, test_funcs]
    Y_valid = np.asarray(Annot['valid_annots'])[:, test_funcs]
    Y_test = np.asarray(Annot['test_annots'])[:, test_funcs]

    mapping_dict = get_mapping_dict(mapping_fname)
    print('Y_train before: ', Y_train.shape)
    annot_prots_train = Annot['prot_IDs'][train_inds]
    annot_prots_train, Y_train = remove_missing_annot_prots(annot_prots_train, Y_train, mapping_dict)
    print('Y_train after: ', Y_train.shape)

    print('Y_valid before: ', Y_valid.shape)
    annot_prots_valid = Annot['prot_IDs'][valid_inds]
    annot_prots_valid, Y_valid = remove_missing_annot_prots(annot_prots_valid, Y_valid, mapping_dict)
    print('Y_valid after: ', Y_valid.shape)

    print('Y_test before: ', Y_test.shape)
    annot_prots_test = Annot['prot_IDs'][test_inds]
    annot_prots_test, Y_test = remove_missing_annot_prots(annot_prots_test, Y_test, mapping_dict)
    print('Y_test after: ', Y_test.shape)
    goterms = Annot['GO_IDs'][test_funcs]

    use_orig_feats = False
    if use_orig_feats:
        # creating a block matrix
        print ("### Creating the block matrix...")
        string_prots = []
        X = [[0]*len(tax_ids) for i in range(len(tax_ids))]
        for ii in range(0, len(tax_ids)):
            Net = pickle.load(open(data_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
            X[ii][ii] = minmax_scale(np.asarray(Net['net'].todense()))
            string_prots += Net['prot_IDs']
            for jj in range(ii + 1, len(tax_ids)):
                R = pickle.load(open(data_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                R = minmax_scale(np.asarray(R.todense()))
                X[ii][jj] = R
                X[jj][ii] = R.T
        X = np.asarray(np.bmat(X))
        print ("### Shape of the block matrix: ", X.shape)
        X = minmax_scale(X)

    else:
        #  Load networks/features
        feature_fname = RESULTS_PATH + model_name.split('-')[0] + '_features.pckl'
        if os.path.isfile(feature_fname):
            print('### Found features in ' + feature_fname + ' Loading it.')
            String = pickle.load(open(feature_fname, 'rb'))
            string_prots = String['prot_IDs']
            X = String['features']
        else:
            '''
            The following code assumes these things:
                - You have the random walk with restart profiles of the string networks already in a pickle file
                - You have the isorank 'block' matrix (rectangle matrix for interspecies connections) (pretty sure 'block' is a misnomer)
            '''
            # creating a block matrix
            print ("### Creating the block matrix...")
            string_prots = []
            X = [[0]*len(tax_ids) for i in range(len(tax_ids))]
            for ii in range(0, len(tax_ids)):
                Net = pickle.load(open(data_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
                X[ii][ii] = minmax_scale(np.asarray(Net['net'].todense()))
                string_prots += Net['prot_IDs']
                for jj in range(ii + 1, len(tax_ids)):
                    R = pickle.load(open(data_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                    R = minmax_scale(np.asarray(R.todense()))
                    X[ii][jj] = R
                    X[jj][ii] = R.T
            X = np.asarray(np.bmat(X))
            print ("### Shape of the block matrix: ", X.shape)

            # X = minmax_scale(String['net'].todense())
            # string_prots = String['prot_IDs']

            '''
            Builds and trains the autoencoder and scales the features.
            '''
            input_dims = [X.shape[1]]
            # encode_dims = [2000, 1000, 2000]
            encode_dims = [1000]
            model, history = build_model(X, input_dims, encode_dims, mtype='ae')
            export_history(history, model_name=model_name, kwrd='AE')

            mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

            X = minmax_scale(mid_model.predict(X))
            String = {}
            String['features'] = X
            String['prot_IDs'] = string_prots
            pickle.dump(String, open(feature_fname, 'wb'))

    # get common indices annotations
    annot_idx_train, string_idx_train = get_common_indices(annot_prots_train, string_prots)
    annot_idx_valid, string_idx_valid = get_common_indices(annot_prots_valid, string_prots)
    annot_idx_test, string_idx_test = get_common_indices(annot_prots_test, string_prots)

    # aligned data
    X_train = X[string_idx_train]
    Y_train = Y_train[annot_idx_train]

    X_valid = X[string_idx_valid]
    Y_valid = Y_valid[annot_idx_valid]

    X_test = X[string_idx_test]
    Y_test = Y_test[annot_idx_test]
    
    perf, y_scores = temporal_holdout(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    print('aupr[micro], aupr[macro], F_max, accuracy\n')
    avg_micro = 0.0
    for ii in range(0, len(perf['F1'])):
        print('%0.5f %0.5f %0.5f %0.5f' % (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['F1'][ii], perf['acc'][ii]))
        avg_micro += perf['pr_micro'][ii]
    avg_micro /= len(perf['F1'])
    print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
    print
    use_nn = False
    if use_nn:
        val_type = 'nn_th'
    else:
        val_type = 'svm_th'
    pickle.dump(y_score_trials, open(RESULTS_PATH + model_name + "_goterm_" + val_type + "_perf.pckl", "wb"))


def leave_one_species_out_main(annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, alpha, test_goid_fname, block_matrix_folder='block_matrix_files/', network_folder='network_files/'):
    # need to make: dictionary of species taxa ids to species inds in the X matrix
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    Y = np.asarray(Annot['annot'][ont].todense())
    annot_prots = Annot['prot_IDs']
    goterms = Annot['go_IDs'][ont]

    use_orig_feats = True
    if use_orig_feats:
        print('Using original feats')
        X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder)

    else:
        #  Load networks/features
        feature_fname = RESULTS_PATH + model_name.split('-')[0] + '_features.pckl'
        if os.path.isfile(feature_fname):
            print('### Found features in ' + feature_fname + ' Loading it.')
            String = pickle.load(open(feature_fname, 'rb'))
            string_prots = String['prot_IDs']
            X = String['features']
            cum_num_prot_ids = [0]
            species_string_prots = String['species_prots']
            for ii in range(0, len(tax_ids)):
                cum_num_prot_ids.append(cum_num_prot_ids[ii] + len(species_string_prots[tax_ids[ii]]))
        else:

            # X = minmax_scale(String['net'].todense())
            # string_prots = String['prot_IDs']
            X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder)

            '''
            Builds and trains the autoencoder and scales the features.
            '''
            input_dims = [X.shape[1]]
            # encode_dims = [2000, 1000, 2000]
            encode_dims = [1000]
            model, history = build_model(X, input_dims, encode_dims, mtype='ae')
            export_history(history, model_name=model_name, kwrd='AE')

            mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

            X = minmax_scale(mid_model.predict(X))
            String = {}
            String['features'] = X
            String['prot_IDs'] = string_prots
            String['species_prots'] = species_string_prots
            pickle.dump(String, open(feature_fname, 'wb'))

    # Load features
    # String_ecoli = pickle.load(open(RESULTS_PATH + 'string_dmelanogaster_features.pckl', 'rb'))
    # string_prots_ecoli = String_ecoli['prot_IDs']
    # ecoli_idx, model_org_idx = get_common_indices(string_prots_ecoli, string_prots)
    # string_prots = [string_prots[ii] for ii in model_org_idx]
    # X = X[model_org_idx]
    
    spec_to_spec_inds = {}
    cum_num_prots_in_annots = 0
    for i in range(0, len(tax_ids)):
        tax_id = tax_ids[i]
        species_prots = species_string_prots[tax_id]
        _, species_idx = get_common_indices(annot_prots, species_prots)
        #spec_to_spec_inds[tax_id] = np.arange(cum_num_prots_in_annots, cum_num_prots_in_annots + len(species_idx))
        spec_to_spec_inds[tax_id] = np.array(species_idx) + cum_num_prots_in_annots
        cum_num_prots_in_annots += len(species_idx)
    
    # get common indices annotations
    annot_idx, string_idx = get_common_indices(annot_prots, string_prots)
    net_inds = []
    net_prots = []
    for tax_id in tax_ids:
        net_inds += list(spec_to_spec_inds[tax_id])
        net_prots += species_string_prots[tax_id]
    try:
        assert net_prots == string_prots
    except AssertionError:
        print('net_prots != string_prots. Exiting.')
        exit()
    try:
        string_idx = sorted(string_idx)
        net_inds = sorted(net_inds)
        assert string_idx == net_inds
    except AssertionError:
        print('String_idx')
        print(string_idx[:10])
        print(string_idx[-10:])
        print('net_inds')
        print(net_inds[:10])
        print(net_inds[-10:])
        exit()
    
    # aligned data
    X = X[string_idx]
    Y = Y[annot_idx]
    print('Y.shape')
    print(Y.shape)
    print('Test Y: ' + Y[spec_to_spec_inds[test_tax_id]])

    print('Assertion: all string')
    assert (np.array(string_prots)[string_idx] == np.array(annot_prots)[annot_idx]).all()
    assert cum_num_prots_in_annots == X.shape[0]

    # selected goids
    test_goids = pickle.load(open(test_goid_fname, 'rb'))
    test_funcs = [goterms.index(goid) for goid in test_goids]
    print('Number of nonzeros in Y matrix total:')
    print(np.count_nonzero(Y))
    Y = Y[:, test_funcs]
    print('Number of nonzeros in Y matrix with these test funcs:')
    print(np.count_nonzero(Y))
    output_projection_files(X, Y, model_name, ont, list(test_goids))

    np.where(Y.any(axis=1))[0]
    print(string_prots[string_idx[10]])
    print(np.array(goterms)[np.where(Y[10, :])])

    #use_nn = True
    use_nn = False
    if use_nn:
        #perf, y_score_trials, y_score_pred = cross_validation_nn(X, Y, n_trials=10, X_pred=X_pred_spec)
        #perf, y_score_trials, y_score_pred = cross_validation_nn(X, Y, n_trials=10, X_pred=None)
        print('Use nn is not implemented for leave one species out. Goodbye.') # TODO: leave one species out not implemented for using maxout nns yet
    
    else:
        #perf, y_score_trials, y_score_pred = cross_validation(X, Y, n_trials=10, X_pred=X_pred_spec)
        perf, y_score_trials = leave_one_species_out_val(X, Y, spec_to_spec_inds, test_tax_id)

    '''
    Pred['prot_IDs'] = pred_spec_prots
    Pred['pred_scores'] = y_score_pred
    '''

    print('aupr[micro], aupr[macro], F_max, accuracy\n')
    avg_micro = 0.0
    for ii in range(0, len(perf['F1'])):
        print('%0.5f %0.5f %0.5f %0.5f' % (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['F1'][ii], perf['acc'][ii]))
        avg_micro += perf['pr_micro'][ii]
    avg_micro /= len(perf['F1'])
    print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
    print
    if use_nn:
        val_type = 'nn'
    else:
        val_type = 'svm'
    pickle.dump(y_score_trials, open(RESULTS_PATH + model_name + "_goterm_" + ont + '_' + val_type + "_loso_perf.pckl", "wb"))
    #pickle.dump(Pred, open(RESULTS_PATH + model_name + '_' + pred_taxon + '_' + ont + '_' + val_type + "_preds.pckl", "wb"))


def load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder):
    # creating a block matrix
    print ("### Creating the block matrix...")
    string_prots = []
    cum_num_prot_ids = [0]
    species_string_prots = {}
    Nets = []
    for ii in range(0, len(tax_ids)):
        print('Loading ' + data_folder + network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl")
        Net = pickle.load(open(data_folder + network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
        Nets.append(Net)
        cum_num_prot_ids.append(cum_num_prot_ids[ii] + len(Net['prot_IDs']))
        string_prots += Net['prot_IDs']
        species_string_prots[tax_ids[ii]] = Net['prot_IDs']

    X = np.zeros((cum_num_prot_ids[-1], cum_num_prot_ids[-1]))
    print('Filling up X matrix')
    for ii in range(0, len(tax_ids)):
        Net = Nets[ii]
        X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(Net['net'].todense()))
    for ii in range(0, len(tax_ids)):
        for jj in range(ii + 1, len(tax_ids)):
            print('Loading ' + data_folder + block_matrix_folder  + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl")
            R = pickle.load(open(data_folder + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
            R = minmax_scale(np.asarray(R.todense()))
            X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1]] = R
            X[cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = R.T
    print ("### Shape of the block matrix: ", X.shape)
    return X, string_prots, species_string_prots


def main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_files/', network_folder='network_files/'):
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    Y = np.asarray(Annot['annot'][ont].todense())
    annot_prots = Annot['prot_IDs']
    goterms = Annot['go_IDs'][ont]

    #use_orig_feats = True
    use_orig_feats = False
    if use_orig_feats:
        '''
        # creating a block matrix
        print('Using original feats')
        print ("### Creating the block matrix...")
        string_prots = []
        X = [[0]*len(tax_ids) for i in range(len(tax_ids))]
        for ii in range(0, len(tax_ids)):
            #Net = pickle.load(open(data_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
            Net = pickle.load(open(data_folder + 'network_files/' + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
            X[ii][ii] = minmax_scale(np.asarray(Net['net'].todense()))
            string_prots += Net['prot_IDs']
            for jj in range(ii + 1, len(tax_ids)):
                #R = pickle.load(open(data_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                block_mat_fname = data_folder + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl"
                print('Loading ' + block_mat_fname)
                R = pickle.load(open(block_mat_fname, "rb"))
                R = minmax_scale(np.asarray(R.todense()))
                X[ii][jj] = R
                X[jj][ii] = R.T
        X = np.asarray(np.bmat(X))
        print ("### Shape of the block matrix: ", X.shape)
        X = minmax_scale(X)
        print(X)
        '''
        X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder)

    else:
        #  Load networks/features
        feature_fname = RESULTS_PATH + model_name.split('-')[0] + '_features.pckl'
        if os.path.isfile(feature_fname):
            print('### Found features in ' + feature_fname + ' Loading it.')
            String = pickle.load(open(feature_fname, 'rb'))
            string_prots = String['prot_IDs']
            X = String['features']
        else:
            '''
            The following code assumes these things:
                - You have the random walk with restart profiles of the string networks already in a pickle file
                - You have the isorank 'block' matrix (rectangle matrix for interspecies connections) (pretty sure 'block' is a misnomer)
            '''
            '''
            # creating a block matrix
            print ("### Creating the block matrix...")
            string_prots = []
            X = [[0]*len(tax_ids) for i in range(len(tax_ids))]
            for ii in range(0, len(tax_ids)):
                Net = pickle.load(open(data_folder + 'network_files/' + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
                X[ii][ii] = minmax_scale(np.asarray(Net['net'].todense()))
                string_prots += Net['prot_IDs']
                for jj in range(ii + 1, len(tax_ids)):
                    print('Loading ' + data_folder + 'block_matrix_files/' + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl")
                    R = pickle.load(open(data_folder + 'block_matrix_files/' + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                    R = minmax_scale(np.asarray(R.todense()))
                    X[ii][jj] = R
                    X[jj][ii] = R.T
            X = np.asarray(np.bmat(X))
            print ("### Shape of the block matrix: ", X.shape)
            '''
            '''
            cum_num_prot_ids = [0]
            species_string_prots = {}
            Nets = []
            for ii in range(0, len(tax_ids)):
                print('Loading ' + data_folder + 'network_files/' + tax_ids[ii] + "_rwr_features_string.v10.5.pckl")
                Net = pickle.load(open(data_folder + 'network_files/' + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
                Nets.append(Net)
                cum_num_prot_ids.append(cum_num_prot_ids[ii] + len(Net['prot_IDs']))
                string_prots += Net['prot_IDs']
                species_string_prots[tax_ids[ii]] = Net['prot_IDs']

            X = np.zeros((cum_num_prot_ids[-1], cum_num_prot_ids[-1]))
            print('Filling up X matrix')
            for ii in range(0, len(tax_ids)):
                Net = Nets[ii]
                X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(Net['net'].todense()))
            for ii in range(0, len(tax_ids)):
                #Net = pickle.load(open(data_folder + 'network_files/' + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
                #X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(Net['net'].todense()))
                for jj in range(ii + 1, len(tax_ids)):
                    print('Loading ' + data_folder + block_matrix_folder  + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl")
                    R = pickle.load(open(data_folder + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                    R = minmax_scale(np.asarray(R.todense()))
                    X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1]] = R
                    X[cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = R.T
            print ("### Shape of the block matrix: ", X.shape)
            '''
            X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder)

            # X = minmax_scale(String['net'].todense())
            # string_prots = String['prot_IDs']

            '''
            Builds and trains the autoencoder and scales the features.
            '''
            input_dims = [X.shape[1]]
            # encode_dims = [2000, 1000, 2000]
            encode_dims = [1000]
            model, history = build_model(X, input_dims, encode_dims, mtype='ae')
            export_history(history, model_name=model_name, kwrd='AE')

            mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

            X = minmax_scale(mid_model.predict(X))
            String = {}
            String['features'] = X
            String['prot_IDs'] = string_prots
            String['species_prots'] = species_string_prots
            pickle.dump(String, open(feature_fname, 'wb'))

    # Load features
    # String_ecoli = pickle.load(open(RESULTS_PATH + 'string_dmelanogaster_features.pckl', 'rb'))
    # string_prots_ecoli = String_ecoli['prot_IDs']
    # ecoli_idx, model_org_idx = get_common_indices(string_prots_ecoli, string_prots)
    # string_prots = [string_prots[ii] for ii in model_org_idx]
    # X = X[model_org_idx]

    '''
    The following code assumes that the species that is going to be predicted for
    had features in the X matrix loaded above.
    '''
    '''
    pred_taxon = '9606'
    Pred = {}
    pred_spec_idx = []
    pred_spec_prots = []
    for j, prot in enumerate(string_prots):
        if prot.startswith(pred_taxon):
            pred_spec_idx.append(j)
            pred_spec_prots.append(prot)
    X_pred_spec = X[pred_spec_idx]
    '''

    # get common indices annotations
    annot_idx, string_idx = get_common_indices(annot_prots, string_prots)

    # aligned data
    X = X[string_idx]
    Y = Y[annot_idx]

    # selected goids
    test_goids = pickle.load(open(test_goid_fname, 'rb'))
    test_funcs = [goterms.index(goid) for goid in test_goids]
    print('Number of nonzeros in Y matrix total:')
    print(np.count_nonzero(Y))
    Y = Y[:, test_funcs]
    print('Number of nonzeros in Y matrix with these test funcs:')
    print(np.count_nonzero(Y))
    output_projection_files(X, Y, model_name, ont, list(test_goids))

    #use_nn = True
    use_nn = False
    if use_nn:
        #perf, y_score_trials, y_score_pred = cross_validation_nn(X, Y, n_trials=10, X_pred=X_pred_spec)
        perf, y_score_trials, y_score_pred = cross_validation_nn(X, Y, n_trials=5, X_pred=None)
    else:
        #perf, y_score_trials, y_score_pred = cross_validation(X, Y, n_trials=10, X_pred=X_pred_spec)
        perf, y_score_trials, y_score_pred = cross_validation(X, Y, n_trials=5, X_pred=None)

    '''
    Pred['prot_IDs'] = pred_spec_prots
    Pred['pred_scores'] = y_score_pred
    '''

    print('aupr[micro], aupr[macro], F_max, accuracy\n')
    avg_micro = 0.0
    for ii in range(0, len(perf['F1'])):
        print('%0.5f %0.5f %0.5f %0.5f' % (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['F1'][ii], perf['acc'][ii]))
        avg_micro += perf['pr_micro'][ii]
    avg_micro /= len(perf['F1'])
    print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
    print
    if use_nn:
        val_type = 'nn'
    else:
        val_type = 'svm'
    pickle.dump(y_score_trials, open(RESULTS_PATH + model_name + "_goterm_" + ont + '_' + val_type + "_perf.pckl", "wb"))
    #pickle.dump(Pred, open(RESULTS_PATH + model_name + '_' + pred_taxon + '_' + ont + '_' + val_type + "_preds.pckl", "wb"))


if __name__ == "__main__":
    annot_fname = str(sys.argv[1])
    ont = str(sys.argv[2])
    model_name = str(sys.argv[3])
    data_folder = str(sys.argv[4])
    if data_folder[-1] != '/':
        data_folder += '/'

    tax_ids = str(sys.argv[5])
    alpha = str(sys.argv[6])
    model_name = model_name + '_alpha_' + str(alpha)
    #val = 'th'
    val = 'cv'
    #val = 'loso'

    # tax ids
    tax_ids = tax_ids.split(',')
    if val == 'cv':
        test_goid_fname = str(sys.argv[7])
        print('annot_fname')
        print(annot_fname)
        print('ontology')
        print(ont)
        print('model_name')
        print(model_name)
        print('data_folder')
        print(data_folder)
        print('tax_ids')
        print(tax_ids)
        print('alpha')
        print(alpha)
        print('test_goid_fname')
        print(test_goid_fname)
        main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_rand_init_test_files_no_add/', network_folder='network_files/')
        #main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_blast_init_test_files/')
        #main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_rand_init_test_files_no_add/', network_folder='network_files_no_add/')
        #main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_rand_init_test_files_2/')
        #main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, block_matrix_folder='block_matrix_files/')
        #main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname)
    elif val == 'th':
        uniprot_mapping_fname = str(sys.argv[7])
        temporal_holdout_main(annot_fname, model_name, data_folder, tax_ids, alpha, uniprot_mapping_fname)
    elif val == 'loso':
        print('Leave one species out...')
        test_goid_fname = str(sys.argv[7])
        test_tax_id = str(sys.argv[8])
        args = [annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, test_goid_fname, alpha]
        leave_one_species_out_main(annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, alpha, test_goid_fname, block_matrix_folder='block_matrix_rand_init_test_files/', network_folder='network_files/')
    else:
        print('Wrong validation setting. Must either be th or cv or loso. Set variable \'val\' inside multispecies.py')
