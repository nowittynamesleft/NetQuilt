import matplotlib as mpl
mpl.use('Agg')
import sys
import pickle
import numpy as np
import os.path
# import scipy.io as sio
from scipy import stats, sparse

from deepNF import build_MDA, build_AE, build_denoising_AE, build_denoising_MDA
from validation import (cross_validation, cross_validation_nn, temporal_holdout,
        output_projection_files, leave_one_species_out_val_nn, 
        train_and_predict_all_orgs, one_spec_cross_val)
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import minmax_scale
#from sklearn.preprocessing import maxabs_scale
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import matplotlib.pyplot as plt
import pandas as pd

import argparse
from utils import ensure_dir


BATCH_SIZE = 128
NB_EPOCH = 100
#NB_EPOCH = 1
LR = 0.01

# python multispecies.py annot_fname ont model_name data_folder tax_ids alpha test_go_id_fname
# example for running autoencoder on human and testing on human on the goids chosen from model-org go ids: python multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function human_only /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl

#For running autoencoder on model orgs and testing on human (with alpha=0.6):
#python multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function model_orgs_human_test /mnt/ceph/users/vgligorijevic/PFL/data/string/ 511145,7227,10090,6239,4932,9606 0.6 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl

# For temporal holdout:
# example for running autoencoder on human and testing on human on the goids chosen from model-org go ids: python multispecies.py ../../data/temporal_holdout/long_time_test_human_MFannotation_data.pkl molecular_function human_only_temporal_holdout_impl_test /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0 

# python multispecies.py annot_fname ont model_name data_folder tax_ids alpha test_go_id_fname test_tax_id

'''
def minmax_scale_sparse(X):
    data = np.asarray(X.data/np.take(X.max(axis=0).todense(), X.indices)).squeeze()
    X_scaled = sparse.csr_matrix((data, X.indices, X.indptr))
    return X_scaled
'''


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


def export_history(history, model_name, kwrd, results_path=None):
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
    annot_idx = [annot_prots.index(prot) for prot in common_prots] # annot_idx is the array of indices in the annotation protein list of each protein common to both annotation and string protein lists
    string_idx = [string_prots.index(prot) for prot in common_prots] # same thing for string protein list

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


def leave_one_species_out_main(annot_fname, ont, model_name, data_folder, tax_ids, 
        test_tax_id, test_annot_fname, alpha, test_goid_fname, results_path='./results/test_results/', 
        block_matrix_folder='block_matrix_files/', network_folder='network_files/', num_hyperparam_sets=1,
        use_orig_feats=True, use_nn=True, arch_set=None, save_only=False, isorank_diag=True, version=1):
    # need to make: dictionary of species taxa ids to species inds in the X matrix
    #  Load annotations
    (X_rest, Y_rest, rest_prot_names, test_goids, X_test_species, Y_test_species, 
            test_species_aligned_net_prots, all_string_prots) = process_and_align_matrices(annot_fname,
        ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, 
        results_path=results_path, block_matrix_folder=block_matrix_folder, 
        network_folder=network_folder, use_orig_feats=use_orig_feats, 
        use_nn=use_nn, test_annot_fname=test_annot_fname, left_out_tax_id=test_tax_id, isorank_diag=isorank_diag, version=version)

    if use_nn:
        perf, pred_file = leave_one_species_out_val_nn(X_test_species, Y_test_species, test_species_aligned_net_prots, X_rest, Y_rest, rest_prot_names, test_tax_id, test_goids, model_name, ont, arch_set=arch_set, save_only=save_only, num_hyperparam_sets=num_hyperparam_sets)
        pickle.dump(pred_file, open(results_path + model_name + '_loso_use_nn_' + ont + '_pred_file.pckl', 'wb'))
    else:
        print('SVM loso validation not implemented. Exiting.')
        exit()
        #perf, y_score_trials = leave_one_species_out_val(X, Y, spec_to_spec_inds, test_tax_id)


def predicted_net_cv_main(annot_fname, ont, model_name, data_folder, tax_ids, 
        test_tax_id, test_annot_fname, alpha, test_goid_fname, results_path='./results/test_results/', 
        block_matrix_folder='block_matrix_files/', network_folder='network_files/', num_hyperparam_sets=1,
        use_orig_feats=True, use_nn=True, arch_set=None, save_only=False, isorank_diag=True):
    other_taxa = [tax_id for tax_id in tax_ids if tax_id != test_tax_id]
    X, Y, aligned_net_prots, test_goids = process_and_align_matrices(annot_fname,
        ont, model_name, data_folder, [test_tax_id], alpha, test_goid_fname, 
        results_path=results_path, block_matrix_folder=block_matrix_folder, 
        network_folder=network_folder, use_orig_feats=use_orig_feats, 
        use_nn=use_nn, test_annot_fname=test_annot_fname, left_out_tax_id=test_tax_id, isorank_diag=isorank_diag, other_taxa=other_taxa)
    if use_nn:
        perf, y_score_trials, pred_file = cross_validation_nn(X, Y, 
            aligned_net_prots, test_goids, model_name, ont, n_trials=n_trials, 
            num_hyperparam_sets=num_hyperparam_sets, arch_set=arch_set)
        pickle.dump(pred_file, open(results_path
                + model_name + '_predicted_net_cv_use_nn_' 
                + ont + '_pred_file.pckl', 'wb'))

def load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder, alpha, left_out_tax_id=None, isorank_diag=False, other_taxa=None, version=1, lm_only=False):
    # creating a block matrix
    print ("### Creating the block matrix...")
    string_prots = []
    cum_num_prot_ids = [0]
    species_string_prots = {}
    Nets = []
    if other_taxa is None:
        for ii in range(0, len(tax_ids)):
            if tax_ids[ii] == left_out_tax_id:
                other_taxa = tax_ids.copy()
                other_taxa.remove(left_out_tax_id)
                left_out_feat_fname = data_folder + network_folder + tax_ids[ii] + "_leftout_network_using_" + ','.join(other_taxa) + "_string.v11.0.pckl"
                print('Loading ' + left_out_feat_fname)
                Net = pickle.load(open(left_out_feat_fname, "rb"))
            elif isorank_diag:
                print('Loading ' + data_folder + network_folder + tax_ids[ii] + "_networks_string.v11.0.pckl")
                Net = pickle.load(open(data_folder + network_folder + tax_ids[ii] + "_networks_string.v11.0.pckl", "rb"))
            else:
                print('Loading ' + data_folder + network_folder + tax_ids[ii] + "_rwr_features_string.v11.0.pckl")
                Net = pickle.load(open(data_folder + network_folder + tax_ids[ii] + "_rwr_features_string.v11.0.pckl", "rb")) # even if isorank diag is actually used, this file still has the protein ids, and so needs to be inputted, but I should restructure the files so that the isorank diag matrices also have the protein ids. Should fix this later.
            Nets.append(Net)
            cum_num_prot_ids.append(cum_num_prot_ids[ii] + len(Net['prot_IDs']))
            string_prots += Net['prot_IDs']
            species_string_prots[tax_ids[ii]] = Net['prot_IDs']
            print('number of proteins in this tax_id: ' + str(len(species_string_prots[tax_ids[ii]])))
    else:
        # test predicted matrix
        left_out_feat_fname = data_folder + network_folder + left_out_tax_id + "_leftout_network_using_" + ','.join(other_taxa) + '_version_' + str(version) + "_string.v11.0.pckl"
        print('Loading ' + left_out_feat_fname)
        Net = pickle.load(open(left_out_feat_fname, "rb"))
        Nets.append(Net)
        cum_num_prot_ids.append(len(Net['prot_IDs']))
        string_prots += Net['prot_IDs']
        species_string_prots[left_out_tax_id] = Net['prot_IDs']
        print('number of proteins in this tax_id: ' + str(len(species_string_prots[left_out_tax_id])))
    if lm_only:
        return None, string_prots, species_string_prots
    X = np.zeros((cum_num_prot_ids[-1], cum_num_prot_ids[-1]))
    #X = sparse.csr_matrix((cum_num_prot_ids[-1], cum_num_prot_ids[-1]))
    print('Filling up X matrix')
    for ii in range(0, len(tax_ids)):
        if isorank_diag and tax_ids[ii] == left_out_tax_id:
            leave_out_block_file = data_folder + block_matrix_folder  + left_out_tax_id + "-" + left_out_tax_id + "_left_out_using_" + ','.join(other_taxa) + '_version_' + str(version) + "_alpha_" + str(alpha) + "_block_matrix.pckl"
            print('Loading ' + leave_out_block_file)
            S_ii = pickle.load(open(leave_out_block_file, 'rb'))
            X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(S_ii.todense()))
        elif isorank_diag:
            diag_block = data_folder + block_matrix_folder  + tax_ids[ii] + "-" + tax_ids[ii] + "_alpha_" + str(alpha) + "_block_matrix.pckl"
            print('Loading ' + diag_block)
            S_ii = pickle.load(open(diag_block, 'rb'))
            X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(S_ii.todense()))
        else:
            Net = Nets[ii]
            X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = minmax_scale(np.asarray(Net['net'].todense()))
            #X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = np.asarray(minmax_scale_sparse(Net['net']))
    for ii in range(0, len(tax_ids)):
        for jj in range(ii + 1, len(tax_ids)):
            if tax_ids[jj] == left_out_tax_id:
                leave_out_block_file = data_folder + block_matrix_folder  + tax_ids[ii] + "-" + tax_ids[jj] + "-leaveout_alpha_" + str(alpha) + "_block_matrix.pckl"
                print('Loading ' + leave_out_block_file)
                R = pickle.load(open(leave_out_block_file, "rb"))
            elif tax_ids[ii] == left_out_tax_id:
                leave_out_block_file = data_folder + block_matrix_folder  + tax_ids[jj] + "-" + tax_ids[ii] + "-leaveout_alpha_" + str(alpha) + "_block_matrix.pckl"
                print('Loading ' + leave_out_block_file)
                R = pickle.load(open(data_folder + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
            else:
                print('Loading ' + data_folder + block_matrix_folder  + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl")
                R = pickle.load(open(data_folder + block_matrix_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))

            R = minmax_scale(np.asarray(R.todense()))
            #R = np.asarray(minmax_scale_sparse(R))
            X[cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1], cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1]] = R
            X[cum_num_prot_ids[jj]:cum_num_prot_ids[jj+1], cum_num_prot_ids[ii]:cum_num_prot_ids[ii+1]] = R.T
    sparsity = 1.0 - ( np.count_nonzero(X) / float(X.size))
    print ("### Sparsity of the block matrix: ", str(sparsity))
    print ("### Shape of the block matrix: ", X.shape)
    print('Length of string_prots')
    print(len(string_prots))
    return X, string_prots, species_string_prots


def predict_main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, results_path='./results/test_results', block_matrix_folder='block_matrix_files/', network_folder='network_files/', arch_set=None, test_set_id_list=None):
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    Y = np.asarray(Annot['annot'][ont].todense())
    annot_prots = Annot['prot_IDs']
    goterms = Annot['go_IDs'][ont]

    print('Using orig features')
    X_to_pred, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder, alpha)
    if test_set_id_list is not None:
        # TODO 
        to_pred_inds = []
        for prot in test_set_id_list:
            to_pred_inds.append(string_prots.index(prot))
        to_pred_inds = np.array(to_pred_inds)
        X_test_list = X_to_pred[to_pred_inds, :]
        

    X_to_pred, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder, alpha, isorank_diag=isorank_diag)

    # get common indices annotations
    annot_idx, string_idx = get_common_indices(annot_prots, string_prots)

    # aligned data
    X = X_to_pred[string_idx]
    Y = Y[annot_idx]

    aligned_net_prots = np.array(string_prots)[string_idx] # don't actually need this to save predictions for ALL string prots

    # selected goids
    test_goids = pickle.load(open(test_goid_fname, 'rb'))
    test_funcs = [goterms.index(goid) for goid in test_goids]
    print('Number of nonzeros in Y matrix total:')
    print(np.count_nonzero(Y))
    Y = Y[:, test_funcs]
    print('Number of nonzeros in Y matrix with these test funcs:')
    print(np.count_nonzero(Y))
    
    if test_set_id_list is not None:
        pred_file = train_and_predict_all_orgs(X, Y, X_test_list, test_set_id_list, test_goids, model_name, ont, arch_set=arch_set)
    else:
        pred_file = train_and_predict_all_orgs(X, Y, X_to_pred, string_prots, test_goids, model_name, ont, arch_set=arch_set)
    pickle.dump(pred_file, open(results_path + model_name + '_use_nn_' + ont + '_pred_file_complete.pckl', 'wb'))


def load_language_model_embeddings(lm_feat_path, tax_ids):
    all_prot_ids = []
    all_embeddings = []
    for tax_id in tax_ids:
        tax_fname = lm_feat_path + '/' + tax_id + '_language_model_features.pckl'
        curr_lm_embedding_dict = pickle.load(open(tax_fname,'rb'))
        prot_ids = curr_lm_embedding_dict['prot_ids']
        embeddings = curr_lm_embedding_dict['features']
        all_prot_ids.extend(prot_ids)
        all_embeddings.append(embeddings)
    return all_prot_ids, np.concatenate(all_embeddings, axis=0)


def process_and_align_matrices(annot_fname, ont, model_name, data_folder, tax_ids, alpha, 
        test_goid_fname, results_path='./results/test_results', block_matrix_folder='block_matrix_files/', 
        network_folder='network_files/', use_orig_feats=False, use_nn=False, test_annot_fname=None, 
        left_out_tax_id=None, isorank_diag=False, other_taxa=None, version=1, lm_feat_path=None,
        lm_only=False):
    '''
    Returns aligned X and Y matrices from annotation filename and tax_id list for networks.
    If test_annot_fname is specified, returns aligned X_test_species and Y_test_species matrices as well, with X and Y not including that species.
    '''
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    Y = np.asarray(Annot['annot'][ont].todense())
    #Y = np.asarray(Annot['annot'][ont])
    annot_prots = Annot['prot_IDs']
    goterms = Annot['go_IDs'][ont]
    if test_annot_fname is not None:
        # need to remove all of the test annotations from Y_rest
        test_Annot = pickle.load(open(test_annot_fname, 'rb'))
        Y_test_species = np.asarray(test_Annot['annot'][ont].todense())
        test_species_annot_prots = test_Annot['prot_IDs']
        print('Y_test_species shape')
        print(Y_test_species.shape)
        print('test_species_annot_prots length:')
        print(len(test_species_annot_prots))
        test_goterms = test_Annot['go_IDs'][ont]
        print('Removing prots from annot prots that are already in test_species_annot_prots')
        print('Before Y shape[0]:')
        print(Y.shape[0])
        print('Before annot_prots len:')
        print(len(annot_prots))
        kept_annot_prots = []
        annot_prot_keep_inds = []
        test_spec_prot_set = set(test_species_annot_prots)
        for i, prot in enumerate(annot_prots):
            if prot not in test_spec_prot_set:
                annot_prot_keep_inds.append(i)
                kept_annot_prots.append(prot)
        annot_prot_keep_inds = np.array(annot_prot_keep_inds)
        Y = Y[annot_prot_keep_inds,:]
        annot_prots = kept_annot_prots
        print('Intersection should be none:')
        print(test_spec_prot_set.intersection(annot_prots))
        print('After annot_prots len:')
        print(len(annot_prots))
        print('After Y shape[0]:')
        print(Y.shape[0])
        assert len(test_species_annot_prots) == Y_test_species.shape[0]

    assert len(annot_prots) == Y.shape[0]
    if use_orig_feats:
        print('Using orig features')
        X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder, alpha, left_out_tax_id=left_out_tax_id, isorank_diag=isorank_diag, other_taxa=other_taxa, version=version, lm_only=lm_only)
        if lm_feat_path is not None:
            lm_prots, lm_embeds = load_language_model_embeddings(lm_feat_path, tax_ids)
            lm_idx = []
            for prot in string_prots:
                assert prot in lm_prots
                lm_idx.append(lm_prots.index(prot))
            lm_idx = np.array(lm_idx)
            lm_prots = np.array(lm_prots)[lm_idx]
            lm_embeds = lm_embeds[lm_idx]
            assert np.all(lm_prots == np.array(string_prots))
            if lm_only:
                X = lm_embeds
            else:
                X = np.concatenate([X, lm_embeds], axis=1)

    else:
        #  Load networks/features
        feature_fname = results_path + model_name.split('-')[0] + '_features.pckl'
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
            X, string_prots, species_string_prots = load_block_mats(data_folder, tax_ids, network_folder, block_matrix_folder, alpha)

            # X = minmax_scale(String['net'].todense())
            # string_prots = String['prot_IDs']

            '''
            Builds and trains the autoencoder and scales the features.
            '''
            input_dims = [X.shape[1]]
            # encode_dims = [2000, 1000, 2000]
            encode_dims = [1000]
            model, history = build_model(X, input_dims, encode_dims, mtype='ae')
            export_history(history, model_name=model_name, kwrd='AE', results_path=results_path)

            mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

            X = minmax_scale(mid_model.predict(X))
            String = {}
            String['features'] = X
            String['prot_IDs'] = string_prots
            String['species_prots'] = species_string_prots
            pickle.dump(String, open(feature_fname, 'wb'))

    '''
    The following code assumes that the species that is going to be predicted for
    had features in the X matrix loaded above.
    '''

    annot_idx, string_idx = get_common_indices(annot_prots, string_prots)
    if test_annot_fname is not None:
        test_annot_idx, test_string_idx = get_common_indices(test_species_annot_prots, string_prots)
        X_test_species = X[test_string_idx] # get indices from big X matrix, because string_prots contains all the proteins from there
        Y_test_species = Y_test_species[test_annot_idx] # get indices from just the annotation matrix whose proteins were inputted in the above get_common_indices call
        test_species_aligned_net_prots = np.array(string_prots)[test_string_idx]

    # aligned data
    X = X[string_idx, :]
    Y = Y[annot_idx, :]
    print(X.shape)
    print(Y.shape)

    aligned_net_prots = np.array(string_prots)[string_idx] # names of proteins that are in X after getting common indices

    # selected goids
    test_goids = pickle.load(open(test_goid_fname, 'rb')) # goids are the same for either setting ("specified test_annot_fname" setting and "unspecified" setting)
    print("Test go ids:")
    print(test_goids)
    print(len(test_goids))
    test_funcs = [goterms.index(goid) for goid in test_goids]
    print("Test funcs:")
    print(test_funcs)
    print(len(test_funcs))
    print('Number of nonzeros in Y matrix total:')
    print(np.count_nonzero(Y))
    Y = Y[:, test_funcs]
    print("Y shape in process_and_align function:")
    print(Y.shape)
    print('Number of nonzeros in Y matrix with these test funcs:')
    print(np.count_nonzero(Y))
    print('Length of string_prots ' + str(len(string_prots)) + ' X.shape[1]: ' + str(X.shape[1]))
    if lm_feat_path is None:
        assert len(string_prots) == X.shape[1]
    if test_annot_fname is not None:
        test_species_test_funcs = [test_goterms.index(goid) for goid in test_goids]
        Y_test_species = Y_test_species[:, test_species_test_funcs]
        return X, Y, aligned_net_prots, test_goids, X_test_species, Y_test_species, test_species_aligned_net_prots, string_prots # string prots should still have the indices of the columns of X
    else:
        return X, Y, aligned_net_prots, test_goids


def main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, test_annot_fname=None, 
        results_path='./results/test_results', block_matrix_folder='block_matrix_files/', 
        network_folder='network_files/', use_orig_feats=False, use_nn=False, 
        num_hyperparam_sets=None, arch_set=None, n_trials=5, save_only=False, load_fname=None,
        isorank_diag=False, subsample=False, lm_feat_path=None, lm_only=False):
    if load_fname is None:
        if test_annot_fname is None:
            X, Y, aligned_net_prots, test_goids = process_and_align_matrices(annot_fname,
                ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, 
                results_path=results_path, block_matrix_folder=block_matrix_folder, 
                network_folder=network_folder, use_orig_feats=use_orig_feats, 
                use_nn=use_nn, test_annot_fname=test_annot_fname, isorank_diag=isorank_diag, lm_feat_path=lm_feat_path, lm_only=lm_only)
        else:
            (X_rest, Y_rest, rest_prot_names, test_goids, X_test_species, Y_test_species, 
                test_species_aligned_net_prots, all_string_prots) = process_and_align_matrices(annot_fname,
                ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, 
                results_path=results_path, block_matrix_folder=block_matrix_folder, 
                network_folder=network_folder, use_orig_feats=use_orig_feats, 
                use_nn=use_nn, test_annot_fname=test_annot_fname, isorank_diag=isorank_diag, lm_feat_path=lm_feat_path, lm_only=lm_only)
    else:
        load_file = pickle.load(open(load_fname, 'rb')) 
        if test_annot_fname is None:
            X = load_file['X']
            Y = load_file['y']
            aligned_net_prots = load_file['prot_names']
            test_goids = load_file['test_goids']
        else:
            X_rest = load_file['X_rest']
            X_test_species = load_file['X_test_species']
            Y_rest = load_file['y_rest']
            Y_test_species = load_file['y_test_species']
            rest_prot_names = load_file['rest_prot_names']
            test_species_aligned_net_prots = load_file['test_species_prots']
            test_goids = load_file['test_goids']

    #print("Saving X and Y matrices") # TODO so I can use a DataGenerator in order to train the maxout nns without loading whole dataset in memory
    # But honestly, not that bad for now
    '''
    trial_file = {}
    if test_annot_fname is None:
        trial_file['X'] = X
        trial_file['Y'] = Y
        trial_file['aligned_net_prots'] = aligned_net_prots
        trial_file['test_goids'] = test_goids
        pickle.dump(trial_file, open('./train_test_data/' + model_name + '_' + ont + '_train_test_data_file.pckl', 'wb'), protocol=4)
    else:
        trial_file['X_rest'] = X_rest
        trial_file['Y_rest'] = Y_rest
        trial_file['rest_prot_names'] = rest_prot_names
        trial_file['test_goids'] = test_goids
        trial_file['X_test_species'] = X_test_species
        trial_file['Y_test_species'] = Y_test_species
        trial_file['test_species_aligned_net_prots'] = test_species_aligned_net_prots
        pickle.dump(trial_file, open('./train_test_data/' + model_name + '_' + ont + '_one_spec_train_test_data_file.pckl', 'wb'), protocol=4)
    print(test_goids)
    exit()
    '''

    #output_projection_files(X, Y, model_name, ont, list(test_goids))
    # 5 fold cross val
    if use_nn:
        if test_annot_fname is not None:
            perf, y_score_trials, pred_file = one_spec_cross_val(X_test_species, 
                    Y_test_species, test_species_aligned_net_prots, X_rest, Y_rest,
                    rest_prot_names, test_goids, model_name, ont, n_trials=n_trials,
                    num_hyperparam_sets=num_hyperparam_sets, arch_set=arch_set, save_only=save_only,
                    subsample=subsample)
            pickle.dump(pred_file, open(results_path
                    + model_name + '_one_spec_cv_use_nn_' 
                    + ont + '_pred_file.pckl', 'wb'))
        else:
            perf, y_score_trials, pred_file = cross_validation_nn(X, Y, 
                aligned_net_prots, test_goids, model_name, ont, n_trials=n_trials, 
                num_hyperparam_sets=num_hyperparam_sets, arch_set=arch_set)
            pickle.dump(pred_file, open(results_path
                    + model_name + '_cv_use_nn_' 
                    + ont + '_pred_file.pckl', 'wb'))
    else:
        perf, y_score_trials, y_score_pred = cross_validation(X, Y, 
                n_trials=5, X_pred=None)

    print('aupr[micro], aupr[macro], F_max, accuracy\n')
    avg_micro = 0.0
    for ii in range(0, len(perf['F1'])):
        print('%0.5f %0.5f %0.5f %0.5f' 
                % (perf['pr_micro'][ii], perf['pr_macro'][ii], perf['F1'][ii], perf['acc'][ii]))
        avg_micro += perf['pr_micro'][ii]
    avg_micro /= len(perf['F1'])
    print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
    print
    if use_nn:
        val_type = 'nn'
    else:
        val_type = 'svm'
    pickle.dump(y_score_trials, 
            open(results_path + model_name 
                + "_goterm_" + ont + '_' + val_type + "_perf.pckl", "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NetQuilt for protein function prediction')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tax_ids', type=str, help="Taxonomy ids of organisms used to train NetQuilt, comma separated (i.e., 511145,316407,316385,224308,71421,243273 for model bacteria)")
    parser.add_argument('--valid_type', type=str, default='cv', help="Validation. Possible: {'cv', 'loso'}.")
    parser.add_argument('--model_name', type=str, default='final_res', help="Output filename keywords.")
    parser.add_argument('--results_path', type=str, default='./results/test_results', help="Saving results.")
    parser.add_argument('--data_folder', type=str, help="Data folder.")
    parser.add_argument('--alpha', type=float, help="Propagation parameter.")

    # added
    parser.add_argument('--annot', type=str, help="Annotation Filename.")
    parser.add_argument('--ont', type=str, default='molecular_function', help="GO term branch.")
    parser.add_argument('--test_goid_fname', type=str, default=None, help="Pickle file containing a list of GO terms to test on. (CV and LOSO valid types only)")
    parser.add_argument('--test_annot_fname', type=str, default=None, help="Optional; additional annotation filename to select cross validation over (leaving out test proteins from this set), training on all annotations given by the --annot filename and testing on 1/5 of the proteins from this annotation file. For this, the species should not be included in the annotations given by the --annot argument (but included in the tax_id list for getting all the RWR/IsoRank features). In loso validation, this would be the left out species' annotation file.")
    parser.add_argument('--test_tax_id', type=str, default=None, help="Taxonomy ID to test on. LOSO valid type only.")
    parser.add_argument('--use_orig_features', help="Use ISORANK S matrix as features for func pred instead of autoencoder features", action='store_true')
    parser.add_argument('--isorank_diag', help="Use ISORANK matrix on diagonal (PPI + BLAST instead of just RWR PPI)", action='store_true')
    parser.add_argument('--use_nn_val', help="Use neural net instead of svm for func pred validation", action='store_true')
    parser.add_argument('--save_only', help="Only create features/associated labels/trial splits for use in cross validation (one spec only for now)", action='store_true')
    parser.add_argument('--num_hyperparam_sets', type=int, help="For using neural networks on original features, gives number of models to train in the hyperparameter search.")
    parser.add_argument('--arch_set', type=str, help="What architecture hyperparam sets to search through for using neural networks on original features (accepted values are for bacteria ('bac') or eukaryotes ('euk'))")
    parser.add_argument('--n_trials', type=int, default=5, help="Number of trials for cv")
    parser.add_argument('--version', type=int, default=5, help="Version of leftout matrix to load (only in loso validation)")
    parser.add_argument('--subsample', help="Randomly sample one-spec-cross-val to have same amount of training samples as only having one species. Used to measure effect of diversity of training examples on performance.", action='store_true')
    parser.add_argument('--lm_feat_path', type=str, default=None, help="Language model feature path.")
    parser.add_argument('--lm_only', action='store_true', default=False, help="Only use language model features.")
    parser.add_argument('--block_mat_folder', type=str, default='block_matrix_ones_init_test_files_no_add/', help="IsoRank block matrices path.")
    parser.add_argument('--net_folder', type=str, default='network_files_no_add/', help="Network pickle file path.")
    parser.add_argument('--test_id_list', type=str, default=None, help="List of STRING ids to predict on as a test set (only for --list_pred validation option).")


    args = parser.parse_args() 

    results_path = args.results_path
    ensure_dir(results_path)
    num_hyperparam_sets = args.num_hyperparam_sets
    annot_fname = args.annot
    ont = args.ont
    model_name = args.model_name
    data_folder = args.data_folder
    if data_folder[-1] != '/':
        data_folder += '/'

    tax_ids = args.tax_ids
    alpha = args.alpha
    model_name = model_name + '_alpha_' + str(alpha)
    val = args.valid_type
    n_trials = args.n_trials

    # tax ids
    tax_ids = tax_ids.split(',')
    test_goid_fname = args.test_goid_fname
    test_tax_id = args.test_tax_id
    use_orig_features = args.use_orig_features
    use_nn = args.use_nn_val
    arch_set = args.arch_set
    test_annot_fname = args.test_annot_fname
    save_only = args.save_only
    isorank_diag = args.isorank_diag
    subsample = args.subsample

    net_folder = args.net_folder
    block_mat_folder = args.block_mat_folder
    #block_mat_folder = 'block_matrix_ones_init_test_files_no_add/'
    #block_mat_folder = 'block_matrix_test_folder/'
    print(args)

    if val == 'cv':
        main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, test_annot_fname=test_annot_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, use_orig_feats=use_orig_features, use_nn=use_nn, num_hyperparam_sets=num_hyperparam_sets, arch_set=arch_set, n_trials=n_trials, save_only=save_only, isorank_diag=isorank_diag, subsample=subsample, lm_feat_path=args.lm_feat_path, lm_only=args.lm_only)
    elif val == 'loso':
        try:
            assert test_tax_id != None
        except AssertionError:
            print('Need to specify tax id to test on for LOSO validation.')
        print('Leave one species out...')
        args = [annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, test_goid_fname, alpha]
        leave_one_species_out_main(annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, test_annot_fname, alpha, test_goid_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, use_orig_feats=use_orig_features, use_nn=use_nn, arch_set=arch_set, save_only=save_only, num_hyperparam_sets=num_hyperparam_sets, isorank_diag=isorank_diag)
    elif val == 'predicted_net_cv':
        predicted_net_cv_main(annot_fname, ont, model_name, data_folder, tax_ids, test_tax_id, test_annot_fname, alpha, test_goid_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, use_orig_feats=use_orig_features, use_nn=use_nn, arch_set=arch_set, save_only=save_only, num_hyperparam_sets=num_hyperparam_sets, isorank_diag=isorank_diag)
    elif val == 'full_prediction':
        print('Full prediction setting. Training on all annotated proteins given, predicting on all proteins given.')
<<<<<<< HEAD
        predict_main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, arch_set=arch_set)
    elif val == 'list_pred':
        print('Training on all annotated proteins of selected taxa that are not in test set ID list.')
        predict_main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, arch_set=arch_set, test_set_id_list=args.test_set_id_list)
=======
        predict_main(annot_fname, ont, model_name, data_folder, tax_ids, alpha, test_goid_fname, results_path=results_path, block_matrix_folder=block_mat_folder, network_folder=net_folder, arch_set=arch_set, isorank_diag=isorank_diag)
>>>>>>> 2b7fe178f674e612fff285ca36d9000becc9f89f
    else:
        print('Wrong validation setting. Must either be cv, loso, full_prediction, or list_pred (with list of string IDs provided in the --test_id_list argument).')
    '''
    elif val == 'th':
        uniprot_mapping_fname = str(sys.argv[7])
        temporal_holdout_main(annot_fname, model_name, data_folder, tax_ids, alpha, uniprot_mapping_fname)
    '''
