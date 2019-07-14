import sys
import pickle
import numpy as np
import os.path
# import scipy.io as sio
from scipy import stats

from deepNF import build_MDA, build_AE
from validation import cross_validation, cross_validation_nn, temporal_holdout, train_test
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sim_reg_utils import build_sim_reg_AE, train_sim_reg_model_given_batches, create_sim_reg_batches_with_unlabeled
from multispecies import build_model

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


BATCH_SIZE = 128
NB_EPOCH = 100
#LR = 0.01 this doesn't do anything, define learning rate in model function
RESULTS_PATH = '../results/test/'
SIM_REG_LAMB = float(sys.argv[-1])
print(SIM_REG_LAMB)

# python multispecies.py annot_fname ont model_name network_folder tax_ids alpha test_go_id_fname
# example for running autoencoder on human and testing on human on the goids chosen from model-org go ids: python sim_reg_multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function human_only_sim_reg /mnt/ceph/users/vgligorijevic/PFL/data/string/ 9606 1.0 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl 0.0

#For running autoencoder on model orgs and testing on human (with alpha=0.6):
#python sim_reg_multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606_string.04_2015_annotations.pckl molecular_function model_orgs_human_test /mnt/ceph/users/vgligorijevic/PFL/data/string/ 511145,7227,10090,6239,4932,9606 0.6 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl 0.0

# For running autoencoder on yeast and testing on yeast
# python sim_reg_multispecies.py /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/4932_string.04_2015_annotations.pckl molecular_function yeast_only_sim_reg_0.0 /mnt/ceph/users/vgligorijevic/PFL/data/string/ 4932 1.0 /mnt/ceph/users/vgligorijevic/PFL/data/annot/string_annot/9606-model-org_molecular_function_train_goids.pckl 0.0


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
    plt.plot(history['recon_loss'], '-')
    plt.plot(history['recon_val_loss'], '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_recon', 'val_recon'], loc='upper left')
    plt.savefig(results_path + model_name.split('-')[0] + '_' + kwrd + '_recon_loss.png', bbox_inches='tight')
    plt.figure()
    plt.plot(history['sim_loss'], '-')
    print('Not including sim val loss in history, because I am assuming they are all 0s')
    #plt.plot(history['sim_val_loss'], '-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    #plt.legend(['train_sim', 'val_sim'], loc='upper left')
    plt.legend(['train_sim'], loc='upper left')
    plt.savefig(results_path + model_name.split('-')[0] + '_' + kwrd + '_sim_loss.png', bbox_inches='tight')


def build_sim_reg_model(X, input_dims, arch, train_inds, test_inds, train_sim_mat, val_sim_mat, labeled_vec_train, labeled_vec_test, nf=0.5, std=1.0, epochs=NB_EPOCH, batch_size=BATCH_SIZE, sim_reg_lamb=SIM_REG_LAMB):
    sim_reg_model = build_sim_reg_AE(input_dims[0], arch, sim_reg_lamb, hidden_activation='tanh')
    # corrupting the input
    noise_factor = nf
    if isinstance(X, list):
        print('Multiple Xs')
        Xs = []
        for net in X:
            Xs.append(net[train_inds])
            Xs.append(net[test_inds])
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
        print('One X')
        X_train = X[train_inds]
        X_test = X[test_inds]
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor*np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)
    # Fitting the model
    history = {}
    print('Creating sim reg batches')
    train_epoch_in_list, train_epoch_out_list, val_epoch_in_list, val_epoch_out_list = create_sim_reg_batches_with_unlabeled(train_sim_mat, val_sim_mat, X_train_noisy, X_train, X_test_noisy,  X_test, labeled_vec_train, labeled_vec_test, batch_size=BATCH_SIZE, num_epochs=NB_EPOCH)
    sim_loss_history, sim_val_loss_history, recon_loss_history, recon_val_loss_history = train_sim_reg_model_given_batches(
                                                                                        sim_reg_model, train_epoch_in_list, 
                                                                                        train_epoch_out_list, val_epoch_in_list, 
                                                                                        val_epoch_out_list, sim_reg_lamb)
    history['sim_loss'] = sim_loss_history
    history['sim_val_loss'] = sim_val_loss_history
    history['recon_loss'] = recon_loss_history
    history['recon_val_loss'] = recon_val_loss_history

    return sim_reg_model, history


def get_common_indices(annot_prots, string_prots):
    common_prots = list(set(string_prots).intersection(annot_prots))
    print ("### Number of prots in intersection:", len(common_prots))
    annot_idx = [annot_prots.index(prot) for prot in common_prots]
    string_idx = [string_prots.index(prot) for prot in common_prots]

    return annot_idx, string_idx

'''
def get_aligned_mats(X, Y, string_idx, annot_idx):
    string_mask = np.zeros(X.shape[0], dtype=bool)
    string_mask[string_idx] = True

    # split X into aligned and unaligned
    X_aligned = X[string_mask, :]
    X_unaligned = X[~string_mask, :]
    Y_aligned = Y[annot_idx]
    return X_aligned, X_unaligned, Y_aligned
'''


def get_aligned_mats(X, Y, annot_prots, string_prots):
    annot_idx, string_idx = get_common_indices(annot_prots, string_prots)

    # split X into aligned and unaligned
    X_aligned = X[string_idx]
    not_string_idx = []
    for i in range(0, X.shape[0]):
        if i not in string_idx:
            not_string_idx.append(i)
    not_string_idx = np.array(not_string_idx)
    X_unaligned = X[not_string_idx]
    Y_aligned = Y[annot_idx]
    return X_aligned, X_unaligned, Y_aligned

def get_annotated_mats(X, Y):
    nonzero_annot_row_inds = np.where(Y.sum(axis=1) != 0)[0]
    Y_annotated = Y[nonzero_annot_row_inds] 

    annotated_mask = np.zeros(X.shape[0], dtype=bool)
    annotated_mask[nonzero_annot_row_inds] = True

    X_annotated = X[annotated_mask]
    X_unannotated = X[~annotated_mask]

    return X_annotated, X_unannotated, Y_annotated, nonzero_annot_row_inds


def get_sim_mats(Y_annotated, train_inds, total_sample_size):
    # calculating pairwise equality of binary labels, converting each row of binary digits to decimal in order to calculate equality
    # only get the train similarity matrix for now
    y_train_with_zero_rows = np.zeros_like(Y_annotated)
    y_train_with_zero_rows[train_inds, :] = np.copy(Y_annotated[train_inds, :])

    print('COSINE SIMS -- NON BINARY')
    train_sim_mat = np.array(cosine_similarity(y_train_with_zero_rows), dtype=np.float32)
    #print('JACCARD SIMS -- NON BINARY')
    #jaccard_distance = pairwise_distances(y_train_with_zero_rows, metric='jaccard')
    #train_sim_mat = 1 - jaccard_distance

    print('zero out the test inds to use for training')
    train_zero_inds = np.where(y_train_with_zero_rows.sum(axis=1) == 0)[0] # includes both removed annotations and rows that were 0 to begin with
    train_sim_mat[train_zero_inds,:] = 0
    train_sim_mat[:, train_zero_inds] = 0 # zero out so that the unannotated proteins are not considered similar to each other

    train_labeled_inds = np.where(y_train_with_zero_rows.sum(axis=1) != 0)[0]

    total_sim_mat_train = np.zeros((total_sample_size, total_sample_size), dtype=np.float32)

    print('Filling big sim mat')
    all_uniprot_inds = np.arange(Y_annotated.shape[0])
    print('All uniprot inds')
    print(all_uniprot_inds.shape)
    all_uniprot_list_inds = [[ind] for ind in all_uniprot_inds] # weird numpy array index assignment (https://stackoverflow.com/questions/30917753/subsetting-a-2d-numpy-array)

    total_sim_mat_train[all_uniprot_list_inds, all_uniprot_inds] = train_sim_mat # train sim mat includes only training similarities (rest are zero)

    similar_inds = np.where(total_sim_mat_train == 1)
    '''
    print(similar_inds[:10])
    print('First 10 total_sim_mat_train similar pair inds(only diag inds where there are annots)')
    for i in range(0, 10):
        print('Should be 0')
        print(sum(Y_annotated[similar_inds[0][i], :] != Y_annotated[similar_inds[1][i], :]))
        print('How many annots each?')
        print(sum(Y_annotated[similar_inds[0][i], :]))
        print(sum(Y_annotated[similar_inds[1][i], :]))
    '''

    diag_inds = np.array(range(0, total_sim_mat_train.shape[0]))
    total_sim_mat_train[diag_inds, diag_inds] = 1.
    print('Train sim mat big shape:')
    print(total_sim_mat_train.shape)
    print(total_sim_mat_train)
    print('NNZ:')
    print(np.count_nonzero(total_sim_mat_train))
    labeled_vec = np.zeros((total_sample_size, 1))
    labeled_vec[train_labeled_inds, :] = 1. # indexed by Y_aligned, anything after Y_aligned.shape[0] is 0

    return total_sim_mat_train, labeled_vec


def main(annot_fname, ont, model_name, network_folder, tax_ids, alpha, test_goid_fname):
    #  Load annotations
    Annot = pickle.load(open(annot_fname, 'rb'))
    Y = np.asarray(Annot['annot'][ont].todense())
    annot_prots = Annot['prot_IDs']
    goterms = Annot['go_IDs'][ont]

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
            Net = pickle.load(open(network_folder + tax_ids[ii] + "_rwr_features_string.v10.5.pckl", "rb"))
            X[ii][ii] = minmax_scale(np.asarray(Net['net'].todense()))
            string_prots += Net['prot_IDs']
            for jj in range(ii + 1, len(tax_ids)):
                R = pickle.load(open(network_folder + tax_ids[ii] + "-" + tax_ids[jj] + "_alpha_" + str(alpha) + "_block_matrix.pckl", "rb"))
                R = minmax_scale(np.asarray(R.todense()))
                X[ii][jj] = R
                X[jj][ii] = R.T
        X = np.asarray(np.bmat(X))
        print ("### Shape of the block matrix: ", X.shape)


    # selected goids
    test_goids = pickle.load(open(test_goid_fname, 'rb'))
    test_funcs = [goterms.index(goid) for goid in test_goids]
    Y = Y[:, test_funcs]

    # aligned data
    X_aligned, X_unaligned, Y_aligned = get_aligned_mats(X, Y, annot_prots, string_prots)
    '''
    for i in range(0, 10):
        print('This is a common aligned protein: ' + str(aligned_prots[i]))
        print('This should be the same one: ' + str(aligned_prots_string[i]))
        if np.sum(Y_aligned[i,:]) != 0:
            annotation = np.array(test_goids)[np.where(Y_aligned[i, :] == 1)[0]]
            #annotation = np.array(goterms)[np.where(Y_aligned[i, :] == 1)[0]]
            print('This is its partial annotation: ' + str(annotation))
    '''

    # take only nonzero rows of Y and put them in Y_annotated
    X_annotated, X_unannotated, Y_annotated, nonzero_annot_row_inds = get_annotated_mats(X_aligned, Y_aligned)
    '''
    for i in range(0, 10):
        print('This is a common annotated protein: ' + str(annotated_prots[i]))
        print('This should be the same one: ' + str(annotated_prots_string[i]))
        annotation = np.array(test_goids)[np.where(Y_annotated[i, :] == 1)[0]]
        #annotation = np.array(goterms)[np.where(Y_annotated[i, :] == 1)[0]]
        print('This is its partial annotation: ' + str(annotation))
    '''

    # okay, so the above seems fine...what gives?
    n_trials = 5
    y_score_trials = np.zeros((Y.shape[1], n_trials), dtype=np.float)
    trial_perfs = {}
    trial_perfs['pr_micro'] = []
    trial_perfs['pr_macro'] = []
    trial_perfs['F1'] = []
    trial_perfs['acc'] = []
    print('Cross validation')
    print('Num trials: ' + str(n_trials))
    for trial in range(0, n_trials):

        annot_train_inds, annot_test_inds = train_test_split(np.arange(0, Y_annotated.shape[0]), test_size=0.2, random_state=1)
        total_sample_size = X.shape[0]
        total_sim_mat_train, labeled_vec = get_sim_mats(Y_annotated, annot_train_inds, total_sample_size)

        X_total = np.concatenate((X_annotated, X_unannotated, X_unaligned), axis=0)
        assert total_sample_size == X_total.shape[0]

        '''
        print('Is anything labeled after Y_annotated.shape[0] entries in the labeled_vec?')
        print(np.sum(labeled_vec[Y_annotated.shape[0]:, :]))
        print('X total shape:')
        print(X_total.shape)
        print('Labeled vec shape:')
        print(labeled_vec.shape)
        print('label vec sum:')
        print(np.sum(labeled_vec))
        print('total sim mat train shape')
        print(total_sim_mat_train.shape)
        '''
        
        input_dims = [X_total.shape[1]]
        #input_dims = [X.shape[1]]
        encode_dims = [1000]

        autoencoder_train_inds, autoencoder_test_inds = train_test_split(np.arange(0, X_total.shape[0]), test_size=0.2)
        train_sim_mat_auto_train = total_sim_mat_train[autoencoder_train_inds, :]
        train_sim_mat_auto_train = train_sim_mat_auto_train[:, autoencoder_train_inds]
        train_sim_mat_auto_test = np.zeros((autoencoder_test_inds.shape[0], autoencoder_test_inds.shape[0]), dtype=np.float32)
        labeled_vec_test = np.zeros((autoencoder_test_inds.shape[0], 1)) # this should make sim val loss always 0
        labeled_vec_train = labeled_vec[autoencoder_train_inds, :]

        model, history = build_sim_reg_model(X_total, input_dims, encode_dims, autoencoder_train_inds, autoencoder_test_inds, train_sim_mat_auto_train, train_sim_mat_auto_test, labeled_vec_train, labeled_vec_test)
        export_history(history, model_name=model_name, kwrd='sim_reg_AE')
        #model, history = build_model(X_total, input_dims, encode_dims, mtype='ae', epochs=NB_EPOCH, batch_size=BATCH_SIZE)

        mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)
        ae_features_annotated = minmax_scale(mid_model.predict(X_annotated))
        print('Features and Y shape for validation:')
        print(ae_features_annotated.shape)

        # okay, finally, now i can use a train-test version of the cross_validation svm script

        perf, y_scores = train_test(ae_features_annotated, Y_annotated, annot_train_inds, annot_test_inds, ker='rbf')
        y_score_trials[:, trial] = y_scores
        trial_perfs['pr_micro'].append(perf['pr_micro'])
        trial_perfs['pr_macro'].append(perf['pr_macro'])
        trial_perfs['F1'].append(perf['F1'])
        trial_perfs['acc'].append(perf['acc'])

        print('aupr[micro], aupr[macro], F_1, accuracy\n')
        avg_micro = 0.0
        print('%0.5f %0.5f %0.5f %0.5f' % (perf['pr_micro'], perf['pr_macro'], perf['F1'], perf['acc']))
    print
    avg_micro = sum(trial_perfs['pr_micro'])/float(len(trial_perfs['pr_micro']))
    print ("### Average (over trials): m-AUPR = %0.3f" % (avg_micro))
    val_type = 'svm'
    pickle.dump(y_score_trials, open(RESULTS_PATH + model_name + "_goterm_" + ont + '_' + val_type + "_perf.pckl", "wb"))


if __name__ == "__main__":
    annot_fname = str(sys.argv[1])
    ont = str(sys.argv[2])
    model_name = str(sys.argv[3]) + '_' + str(SIM_REG_LAMB)
    print('model name: ' + model_name)
    network_folder = str(sys.argv[4])
    if network_folder[-1] != '/':
        network_folder += '/'

    tax_ids = str(sys.argv[5])
    alpha = str(sys.argv[6])
    test_goid_fname = str(sys.argv[7])

    # tax ids
    tax_ids = tax_ids.split(',')
    main(annot_fname, ont, model_name, network_folder, tax_ids, alpha, test_goid_fname)
