import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, cosine_similarity
from sklearn.utils import resample
from sklearn.metrics.scorer import make_scorer
from keras.models import Model
from keras.layers import Input, Dense, maximum, BatchNormalization, Dropout
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
import talos as ta
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import math_ops
from utils import ensure_dir


import datetime
import os

BAC_PARAMS = {'hidden_dim_1': [500],
                    'hidden_dim_2': [0, 300, 600],
                    'hidden_dim_3': [0, 800],
                    'hidden_dim_4': [800, 1000],
                    'maxout_units': [3],
                    'dropout': [0.2, 0.4],
                    'epochs': [50, 75, 100],
                    #'epochs': [25],
                    #'epochs': [1], # test
                    'learning_rate': [0.01],
                    'activation': ['relu'], # no activation if maxout
                    'batch_size': [16, 32],
                    #'exp_name': [exp_name]
}

BAC_PARAMS_NO_SEARCH = {'hidden_dim_1': [500],
                    'hidden_dim_2': [0],
                    'hidden_dim_3': [800],
                    'hidden_dim_4': [800],
                    'maxout_units': [3],
                    'dropout': [0.2],
                    'epochs': [100],
                    #'epochs': [25],
                    #'epochs': [1], # test
                    'learning_rate': [0.01],
                    'activation': ['maxout'], # specifying this doesn't do anything, but prevents the model from running if "maxout units" is not specified
                    'batch_size': [16],
                    #'exp_name': [exp_name]
}
EUK_PARAMS = {'hidden_dim_1': [500],
            'hidden_dim_2': [0], 
            'hidden_dim_3': [800],
            'hidden_dim_4': [800],
            'maxout_units': [4], 
            'dropout': [0.2],
            'epochs': [300],
            #'epochs': [1],
            'learning_rate': [0.01],
            'activation': ['relu'], # no activation if maxout
            'batch_size': [32],
}

EUK_PARAMS_NO_SEARCH = {'hidden_dim_1': [500],
            'hidden_dim_2': [0], 
            'hidden_dim_3': [800],
            'hidden_dim_4': [800],
            'maxout_units': [4], 
            'dropout': [0.2],
            'epochs': [300],
            #'epochs': [1],
            'learning_rate': [0.01],
            'activation': ['maxout'], # specifying this doesn't do anything, but prevents the model from running if "maxout units" is not specified
            'batch_size': [32],
}

def kernel_func(X, Y=None, param=0):
    if param != 0:
        K = rbf_kernel(X, Y, gamma=param)
    else:
        K = linear_kernel(X, Y)

    return K


def trapezoidal_integral_approx(t, y):
    return math_ops.reduce_sum(
            math_ops.multiply(t[1:] - t[:-1],
                (y[:-1] + y[1:]) / 2.), 
            name='trapezoidal_integral_approx')


def micro_AUPR(label, score):
    """Computing AUPR (micro-averaging)"""
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


def micro_AUPR_tensors(label, score):
    """Computing AUPR for use in keras metrics argument for fit function"""
    label = tf.reshape(label, [-1])
    score = tf.reshape(score, [-1])

    order = tf.argsort(score)[::-1]
    label = tf.gather(label, order)

    P = tf.dtypes.cast(tf.count_nonzero(label), tf.float32)
    # N = len(label) - P

    TP = tf.cumsum(label)
    PP = tf.dtypes.cast(tf.range(1, tf.shape(label)[0]+1), tf.float32)  # python

    x = tf.divide(TP, P)  # recall
    y = tf.divide(TP, PP)  # precision

    pr = trapezoidal_integral_approx(x, y)
    #f = tf.divide(2*x*y, (x + y))
    #idx = tf.where((x + y) != 0)[0]
    #f = tf.cond(tf.not_equal(tf.shape(idx)[0], tf.constant(0)), lambda: tf.reduce_max(tf.gather(f, idx)), lambda: tf.constant(0.0))

    return pr


def micro_AUPR_tensors_packed(packed):
    """Computing AUPR for use in keras metrics argument for fit function"""
    label = packed[0]
    score = packed[1]
    print(label)
    print(score)
    label = tf.reshape(label, [-1])
    score = tf.reshape(score, [-1])

    order = tf.argsort(score)[::-1]
    label = tf.gather(label, order)

    P = tf.dtypes.cast(tf.count_nonzero(label), tf.float32)
    # N = len(label) - P

    TP = tf.cumsum(label)
    PP = tf.dtypes.cast(tf.range(1, tf.shape(label)[0]+1), tf.float32)  # python

    x = tf.divide(TP, P)  # recall
    y = tf.divide(TP, PP)  # precision

    pr = trapezoidal_integral_approx(x, y)
    #f = tf.divide(2*x*y, (x + y))
    #idx = tf.where((x + y) != 0)[0]
    #f = tf.cond(tf.not_equal(tf.shape(idx)[0], tf.constant(0)), lambda: tf.reduce_max(tf.gather(f, idx)), lambda: tf.constant(0.0))

    return pr


def micro_AUPR_macro_tensors(label, score):
    """Computing AUPR for use in keras metrics argument for fit function"""
    #label = tf.reshape(label, [-1])
    #score = tf.reshape(score, [-1])


    #f = tf.divide(2*x*y, (x + y))
    #idx = tf.where((x + y) != 0)[0]
    #f = tf.cond(tf.not_equal(tf.shape(idx)[0], tf.constant(0)), lambda: tf.reduce_max(tf.gather(f, idx)), lambda: tf.constant(0.0))
    label_vecs = tf.split(label, score.shape[1], axis=1)
    score_vecs = tf.split(score, score.shape[1], axis=1)
    pr_vals = tf.map_fn(micro_AUPR_tensors_packed, (label_vecs, score_vecs))
    pr = tf.reduce_mean(pr_vals)

    return pr


def ml_split(y):
    """Split annotations"""
    kf = KFold(n_splits=5, shuffle=True)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def evaluate_performance(y_test, y_score, y_pred):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()

    # Compute macro-averaged AUPR
    perf["pr_macro"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i] = micro_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["pr_macro"] += perf[i]
    perf["pr_macro"] /= n

    # Compute micro-averaged AUPR
    perf["pr_micro"] = micro_AUPR(y_test, y_score)

    # Computes accuracy
    perf['acc'] = accuracy_score(y_test, y_pred)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_pred)
    for i in range(y_pred.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1"] = f1_score(y_test, y_new_pred, average='micro')

    return perf


def remove_zero_annot_rows(X, y):
    keep_rows = np.where(y.sum(axis=1) != 0)[0]
    y = y[keep_rows]
    X = X[keep_rows]

    return X, y


def leave_one_species_out_val_nn(X_test_species, y_test_species, test_species_prots, 
        X_rest, y_rest, rest_prot_names, test_species, go_terms, keyword, ont, num_hyperparam_sets=1,
        arch_set=None, save_only=False):
    print('Commencing leave one species out validation for test_species ' + str(test_species))
    # spec_inds is a dictionary with keys = species taxa ids : values = species indices in 
    # the X matrix and y matrix
    assert X_test_species.shape[0] > 0 and y_test_species.shape[0] > 0
    assert X_rest.shape[0] > 0 and y_rest.shape[0] > 0

    '''
    train_species = [species for species in spec_inds.keys() if species != test_species]
    print('train species:')
    print(train_species)
    test_inds = spec_inds[test_species]
    print(test_inds)

    train_inds = np.delete(np.arange(X.shape[0]), test_inds, axis=0)
    X_train = X[train_inds, :] # cross-val on these to choose hyperparams
    y_train = y[train_inds, :]
    X_test = X[test_inds, :]
    y_test = y[test_inds, :]
    print('X_train shape')
    print(X_train.shape)
    print('y_train shape')
    print(y_train.shape)

    # delete 0 rows
    X_train, y_train = remove_zero_annot_rows(X_train, y_train)
    X_test, y_test = remove_zero_annot_rows(X_test, y_test)
    '''
    print("Shapes of X and Y matrices")
    print('X_test_species')
    print(X_test_species.shape)
    print('X_rest')
    print(X_rest.shape)
    print('y_test_species')
    print(y_test_species.shape)
    print('y_rest')
    print(y_rest.shape)
    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    if save_only:
        data_file = {}
        data_file['X_rest'] = X_rest
        data_file['Y_rest'] = y_rest
        data_file['rest_prot_names'] = rest_prot_names
        data_file['test_goids'] = go_terms
        data_file['X_test_species'] = X_test_species
        data_file['Y_test_species'] = y_test_species
        data_file['test_species_prots'] = test_species_prots
        dump_fname = './train_test_data/' + keyword + '_' + ont + '_loso_train_test_data_file.pckl'
        pickle.dump(data_file, open(dump_fname, 'wb'), protocol=4)
        exit()

    X_test_species, y_test_species, test_species_prots = remove_zero_annot_rows_w_labels(X_test_species, 
            y_test_species, test_species_prots)
    X_rest, y_rest, rest_prots = remove_zero_annot_rows_w_labels(X_rest, y_rest, 
            rest_prot_names)
    it = 0
    pred_file = {'prot_IDs': test_species_prots,
                 'GO_IDs': go_terms,
                 'preds': np.zeros_like(y_test_species),
                 'true_labels': y_test_species,
                 }
    print ("Train samples=%d; #Test samples=%d" % (y_rest.shape[0], y_test_species.shape[0]))
    #downsample_rate = 0.01 # for bacteria
    #downsample_rate = 0.001 # for eukaryotes

    ensure_dir('hyperparam_searches/')
    exp_name = 'hyperparam_searches/' + keyword + '-' + ont + 'loso_val'
    # no architecture search, just run it
    if arch_set == 'bac':
        # for bacteria
        print("RUNNING MODEL ARCHITECTURE FOR BACTERIA")
        params = BAC_PARAMS
        if int(num_hyperparam_sets) == 1:
            print('No search')
            params = BAC_PARAMS_NO_SEARCH
    elif arch_set == 'euk':
        # for eukaryotes
        print("RUNNING MODEL ARCHITECTURES FOR EUKARYOTES")
        params = EUK_PARAMS
        if int(num_hyperparam_sets) == 1:
            print('No search')
            params = EUK_PARAMS_NO_SEARCH
    else:
        print('No arch_set chosen! Need to specify in order to know which hyperparameter sets to search through for cross-validation using neural networks with original features.')

    exp_path = exp_name + '_num_hyperparam_sets_' + str(num_hyperparam_sets)
    #params = {param_name:param_list[0] for (param_name, param_list) in params.items()}
    if num_hyperparam_sets > 1:
        # hyperparam search
        print('number of hyperparam sets to train:' + str(num_hyperparam_sets))
        params['in_shape'] = [X_train.shape[1]]
        params['out_shape'] = [y_train.shape[1]]
        keras_model = KerasClassifier(build_fn=build_maxout_nn_classifier)
        clf = RandomizedSearchCV(keras_model, params, cv=2, n_iter=num_hyperparam_sets, scoring=make_scorer(micro_AUPR, greater_is_better=True)) # this is training on half the training data, should probably not do this
        search_result = clf.fit(X_train, y_train)
        # summarize results
        means = search_result.cv_results_['mean_test_score']
        stds = search_result.cv_results_['std_test_score']
        params = search_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
        best_params = search_result.best_params_
        print('Best model parameters for this trial:')
        print(best_params)
        print ("### Using full training data...")
        with open(exp_path + '_search_results.pckl', 'wb') as search_result_file:
            pickle.dump(search_result, search_result_file)
        del best_params['in_shape']
        del best_params['out_shape']
    else:
        # no hyperparam search
        best_params = {param_name:param_list[0] for (param_name, param_list) in params.items()}

    print ("### Using full training data...")
    print("Using %s" % (best_params))
    best_params['exp_name'] = exp_name
    history, model = build_and_fit_nn_classifier(X_rest, y_rest, best_params, verbose=1)

    y_score = np.zeros(y_test_species.shape, dtype=float)
    y_pred = np.zeros_like(y_test_species)

    # Compute performance on test set
    y_score = model.predict(X_test_species)
    pred_file['preds'] = model.predict(X_test_species)
    y_pred = y_score > 0.5 #silly way to do predictions from the scores; choose threshold, maybe use platt scaling or something else
    perf = evaluate_performance(y_test_species, y_score, y_pred)
    print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf['pr_micro'], perf['pr_macro'], perf['F1'], perf['acc']))
    print
    print

    return perf, pred_file


def train_test(X, y, train_idx, test_idx, ker='rbf'):
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_scores = np.zeros((y.shape[1]), dtype=np.float)
    y_train = y[train_idx]
    y_test = y[test_idx]
    print('Y_train before')
    print(y_train.shape)
    print('Y_test before')
    print(y_test.shape)
    print('X_train before')
    print(X_train.shape)
    print('X_test before')
    print(X_test.shape)

    print('This should do nothing:')
    X_train, y_train = remove_zero_annot_rows(X_train, y_train)
    X_test, y_test = remove_zero_annot_rows(X_test, y_test)
    print('Y_train after')
    print(y_train.shape)
    print('Y_test after')
    print(y_test.shape)
    print('X_train after')
    print(X_train.shape)
    print('X_test after')
    print(X_test.shape)

    # now I have a training and testing X and y with no zero rows.
    # Make kernels for X_train and X_test
    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    if ker == 'rbf':
        gamma_range = 10.**np.arange(-3, 1)
    elif ker == 'lin':
        gamma_range = [0]
    else:
        print ("### Wrong kernel.")

    # pre-generating kernels
    print ("### Pregenerating kernels...")
    K_rbf_train = {}
    K_rbf_test = {}
    for gamma in gamma_range:
        K_rbf_train[gamma] = kernel_func(X_train, param=gamma) # K_rbf_train has the same indices as y_train and X_train
        K_rbf_test[gamma] = kernel_func(X_test, X_train, param=gamma) # K_rbf_test has the same indices as y_test and X_test on axis 0 and X_train on axis 1


    print ("### Done.")

    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))
    # setup for nested cross-validation
    splits = ml_split(y_train)

    # parameter fitting
    C_opt = None
    gamma_opt = None
    max_aupr = 0
    for C in C_range:
        for gamma in gamma_range:
            # Multi-label classification
            cv_results = []
            for train, valid in splits:
                clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                                  random_state=123,
                                                  probability=True), n_jobs=-1)
                K_train = K_rbf_train[gamma][train, :][:, train]
                K_valid = K_rbf_train[gamma][valid, :][:, train]
                y_train_t = y_train[train]
                y_train_v = y_train[valid]
                y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                y_pred_valid = np.zeros_like(y_train_v)
                idx = np.where(y_train_t.sum(axis=0) > 0)[0] # why is this necessary? oh, not all go terms definitely have labels for random splits. gotcha
                '''
                print('Before training:')
                print('K_train shape:')
                print(K_train.shape)
                print('y_train_t shape:')
                print(y_train_t.shape)
                '''
                clf.fit(K_train, y_train_t[:, idx])
                # y_score_valid[:, idx] = clf.decision_function(K_valid)
                y_score_valid[:, idx] = clf.predict_proba(K_valid)
                y_pred_valid[:, idx] = clf.predict(K_valid)
                perf_cv = evaluate_performance(y_train_v,
                                               y_score_valid,
                                               y_pred_valid)
                cv_results.append(perf_cv['pr_micro'])
            cv_aupr = np.median(cv_results)
            print ("### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, cv_aupr))
            if cv_aupr > max_aupr:
                C_opt = C
                gamma_opt = gamma
                max_aupr = cv_aupr
    print ("### Optimal parameters: ")
    print ("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
    print ("### Train dataset: AUPR = %0.3f" % (max_aupr))
    print
    print ("### Using full training data...")
    clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                      random_state=123,
                                      probability=True), n_jobs=-1)
    y_score = np.zeros(y_test.shape, dtype=float)
    y_pred = np.zeros_like(y_test)
    # idx = np.where(y_train.sum(axis=0) > 0)[0]
    clf.fit(K_rbf_train[gamma_opt], y_train)

    # Compute performance on test set
    # y_score[:, idx] = clf.decision_function(K_rbf[gamma_opt][test_idx, :][:, train_idx])
    # y_score[:, idx] = clf.predict_proba(K_rbf[gamma_opt][test_idx, :][:, train_idx])
    # y_pred[:, idx] = clf.predict(K_rbf[gamma_opt][test_idx, :][:, train_idx])
    y_score = clf.predict_proba(K_rbf_test[gamma_opt])
    y_pred = clf.predict(K_rbf_test[gamma_opt])
    perf_trial = evaluate_performance(y_test, y_score, y_pred)
    for go_id in range(0, y_pred.shape[1]):
        y_scores[go_id] = perf_trial[go_id]
    print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['pr_micro'], perf_trial['pr_macro'], perf_trial['F1'], perf_trial['acc']))
    print
    print

    return perf_trial, y_scores




def cross_validation(X, y, n_trials=5, ker='rbf', X_pred=None):
    """Perform model selection via 5-fold cross validation"""
    # filter samples with no annotations
    X, y = remove_zero_annot_rows(X, y)
    print('X and y shape:')
    print(X.shape)
    print(y.shape)
    print('Num nonzeros in y matrix:')
    print(np.count_nonzero(y))

    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    if ker == 'rbf':
        gamma_range = 10.**np.arange(-3, 1)
    elif ker == 'lin':
        gamma_range = [0]
    else:
        print ("### Wrong kernel.")

    # pre-generating kernels
    print ("### Pregenerating kernels...")
    K_rbf = {}
    for gamma in gamma_range:
        K_rbf[gamma] = kernel_func(X, param=gamma)

    if X_pred is not None:
        y_score_pred = np.zeros((X_pred.shape[0], y.shape[1]), dtype=np.float)
        K_rbf_pred = {}
        for gamma in gamma_range:
            K_rbf_pred[gamma] = kernel_func(X_pred, X, param=gamma)

    print ("### Done.")

    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    # shuffle and split training and test sets
    '''
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
    ss = trials.split(X)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))
    '''
    k_fold = KFold(n_splits=n_trials)
    trial_splits = k_fold.split(X, y=y)

    y_score_trials = np.zeros((y.shape[1], n_trials), dtype=np.float)
    it = 0
    for train_idx, test_idx in trial_splits:
        '''
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
        '''
        it += 1
        y_train = y[train_idx]
        y_test = y[test_idx]
        print ("### [Trial %d] Perfom cross validation...." % (it))
        print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))
        # setup for nested cross-validation
        splits = ml_split(y_train)

        # parameter fitting
        C_opt = None
        gamma_opt = None
        max_aupr = 0
        for C in C_range:
            for gamma in gamma_range:
                # Multi-label classification
                cv_results = []
                for train, valid in splits:
                    clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                                      random_state=123,
                                                      probability=True), n_jobs=-1)
                    K_train = K_rbf[gamma][train_idx[train], :][:, train_idx[train]]
                    K_valid = K_rbf[gamma][train_idx[valid], :][:, train_idx[train]]
                    y_train_t = y_train[train]
                    y_train_v = y_train[valid]
                    y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                    y_pred_valid = np.zeros_like(y_train_v)
                    idx = np.where(y_train_t.sum(axis=0) > 0)[0]
                    clf.fit(K_train, y_train_t[:, idx])
                    # y_score_valid[:, idx] = clf.decision_function(K_valid)
                    y_score_valid[:, idx] = clf.predict_proba(K_valid)
                    y_pred_valid[:, idx] = clf.predict(K_valid)
                    perf_cv = evaluate_performance(y_train_v,
                                                   y_score_valid,
                                                   y_pred_valid)
                    cv_results.append(perf_cv['pr_micro'])
                cv_aupr = np.median(cv_results)
                print ("### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, cv_aupr))
                if cv_aupr > max_aupr:
                    C_opt = C
                    gamma_opt = gamma
                    max_aupr = cv_aupr
        print ("### Optimal parameters: ")
        print ("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
        print ("### Train dataset: AUPR = %0.3f" % (max_aupr))
        print
        print ("### Using full training data...")
        clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                          random_state=123,
                                          probability=True), n_jobs=-1)
        y_score = np.zeros(y_test.shape, dtype=float)
        y_pred = np.zeros_like(y_test)
        # idx = np.where(y_train.sum(axis=0) > 0)[0]
        clf.fit(K_rbf[gamma_opt][train_idx, :][:, train_idx], y_train)

        # Compute performance on test set
        # y_score[:, idx] = clf.decision_function(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        # y_score[:, idx] = clf.predict_proba(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        # y_pred[:, idx] = clf.predict(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        y_score = clf.predict_proba(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        y_pred = clf.predict(K_rbf[gamma_opt][test_idx, :][:, train_idx])
        perf_trial = evaluate_performance(y_test, y_score, y_pred)
        for go_id in range(0, y_pred.shape[1]):
            y_score_trials[go_id, it-1] = perf_trial[go_id]
        pr_micro.append(perf_trial['pr_micro'])
        pr_macro.append(perf_trial['pr_macro'])
        F1.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['pr_micro'], perf_trial['pr_macro'], perf_trial['F1'], perf_trial['acc']))
        print
        print
        if X_pred is not None:
            print ("### Predicting functions...")
            y_score_pred += clf.predict_proba(K_rbf_pred[gamma_opt][:, train_idx])
            y_score_pred /= n_trials
        else:
            y_score_pred = []

    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['F1'] = F1
    perf['acc'] = acc

    return perf, y_score_trials, y_score_pred


def build_maxout_nn_classifier_wan(input_dim, output_dim, maxout_units): # this is all the hyperparameters of Wan et al.
    optim = Adagrad(lr=0.05)
    '''
    the following code doesn't work as keras 2.0 removed maxout dense layer
    model = Sequential()
    model.add(MaxoutDense(500, nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxoutDense(700, nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=element[2]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxoutDense(800, nb_feature=3, init='glorot_uniform', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_shape=(len(MatrixFeatures[0]),)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='sigmoid'))

    instead, the following convention is used for maxout layers. 
    max_out = maximum([Dense(neurons, **dense_args)(input_layer) for _ in range(n_pieces)])
    '''
    input_layer = Input(shape=(input_dim,))
    max_out_1 = maximum([Dense(500)(input_layer) for _ in range(maxout_units)]) # 3 pieces
    batch_norm_1 = BatchNormalization()(max_out_1)
    dropout_1 = Dropout(0.5)(batch_norm_1)
    max_out_2 = maximum([Dense(700)(dropout_1) for _ in range(maxout_units)]) # 3 pieces
    batch_norm_2 = BatchNormalization()(max_out_2)
    dropout_2 = Dropout(0.5)(batch_norm_2)
    max_out_3 = maximum([Dense(800)(dropout_2) for _ in range(maxout_units)]) # 3 pieces
    batch_norm_3 = BatchNormalization()(max_out_3)
    dropout_3 = Dropout(0.5)(batch_norm_3)
    output_layer = Dense(output_dim, activation='sigmoid')(dropout_3)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f1_score])
    model.summary()
    return model
   

def build_maxout_nn_classifier(in_shape=0, out_shape=0, hidden_dim_1=500, hidden_dim_2=0, hidden_dim_3=800, hidden_dim_4=800, maxout_units=3, activation='sigmoid', dropout=0.0, learning_rate=0.01):
    K.clear_session()
    input_layer = Input(shape=(in_shape,))
    '''
    hidden_dim_1 = int(params['hidden_dim_1'])
    hidden_dim_2 = int(params['hidden_dim_2']) if params['hidden_dim_2'] != 'None' else None
    hidden_dim_3 = int(params['hidden_dim_3']) if params['hidden_dim_3'] != 'None' else None
    hidden_dim_4 = int(params['hidden_dim_4']) if params['hidden_dim_4'] != 'None' else None
    '''
    if maxout_units == 0:
        maxout = False
    else:
        maxout = True
    x = input_layer
    if hidden_dim_1 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_1)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_1, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    if hidden_dim_2 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_2)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_2, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    if hidden_dim_3 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_3)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_3, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    if hidden_dim_4 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_4)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_4, activation=activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
    output_layer = Dense(out_shape, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    optim = Adagrad(lr=learning_rate)
    #model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f1_score])
    model.compile(optimizer=optim, loss='binary_crossentropy')
    model.summary()
    
    return model 


def create_param_dict_string(params):
    dict_string = params['exp_name'] + '_maxout_' + str(params['maxout_units']) + '_arch_'
    for i in range(1,5):
        dict_string += str(params['hidden_dim_' + str(i)]) + '_'
    if params['maxout_units'] > 0:
        dict_string += 'act_maxout_lr_' + str(params['learning_rate']) + '_num_epoch_' + str(params['epochs']) + '_batch_size_' + str(params['batch_size'])
    else:
        dict_string += 'act_' + params['activation'] + '_lr_' + str(params['learning_rate']) + '_num_epoch_' + str(params['epochs']) + '_batch_size_' + str(params['batch_size'])
    return dict_string
   

#def build_and_fit_nn_classifier(X_train, y_train, X_val, y_val, hidden_dim_1=500, hidden_dim_2=0, hidden_dim_3=800, hidden_dim_4=800, maxout_units=3, activation='sigmoid', dropout=0.0, learning_rate=0.01, batch_size=64, num_epochs=200, exp_name='no_name'):
def build_and_fit_nn_classifier(X, y, params, X_val=None, y_val=None, verbose=0):
    if X_val is None:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
    else:
        X_train = X
        y_train = y
    print(X_train[0][X_train[0] != 0])
    K.clear_session()
    input_layer = Input(shape=(X_train.shape[1],))
    '''
    hidden_dim_1 = int(params['hidden_dim_1'])
    hidden_dim_2 = int(params['hidden_dim_2']) if params['hidden_dim_2'] != 'None' else None
    hidden_dim_3 = int(params['hidden_dim_3']) if params['hidden_dim_3'] != 'None' else None
    hidden_dim_4 = int(params['hidden_dim_4']) if params['hidden_dim_4'] != 'None' else None
    '''
    hidden_dim_1 = int(params['hidden_dim_1'])
    hidden_dim_2 = int(params['hidden_dim_2'])
    hidden_dim_3 = int(params['hidden_dim_3'])
    hidden_dim_4 = int(params['hidden_dim_4'])
    if params['maxout_units'] == 0:
        maxout = False
    else:
        maxout = True
        maxout_units = int(params['maxout_units'])
    x = input_layer
    if hidden_dim_1 != 0:
        if maxout:
            x = maximum([Dense(int(params['hidden_dim_1']))(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_1, activation=params['activation'])(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)
    if hidden_dim_2 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_2)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_2, activation=params['activation'])(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)
    if hidden_dim_3 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_3)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_3, activation=params['activation'])(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)
    if hidden_dim_4 != 0:
        if maxout:
            x = maximum([Dense(hidden_dim_4)(x) for _ in range(maxout_units)]) # 3 pieces
        else:
            x = Dense(hidden_dim_4, activation=params['activation'])(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout'])(x)
    output_layer = Dense(y_train.shape[1], activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    optim = Adagrad(lr=params['learning_rate'])
    #model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[f1_score])
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[micro_AUPR_tensors])
    model.summary()
   
    early = EarlyStopping(monitor='val_micro_AUPR_tensors', mode='max', verbose=1, min_delta=0, patience=30)
    #early = EarlyStopping(monitor='val_loss', mode='min', verbose=1, min_delta=0, patience=30)
    #early = EarlyStopping(monitor='val_micro_AUPR_tensors', mode='max', verbose=1, min_delta=0, patience=5)
    print('Fitting model now:')
    history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=int(params['batch_size']),  epochs=int(params['epochs']), verbose=verbose, callbacks=[early])
    #history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=int(params['batch_size']),  epochs=int(params['epochs']), verbose=verbose)
    y_pred_val = model.predict(X_val)
    micro_val = micro_AUPR(y_val, y_pred_val)
    print('Micro aupr of validation set')
    print(micro_val)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    param_dict_string = create_param_dict_string(params)

    loss_string = param_dict_string + '_loss.png'
    print('Saving ' + loss_string)
    plt.savefig(loss_string)
    plt.close()

    plt.plot(history.history['micro_AUPR_tensors'])
    plt.plot(history.history['val_micro_AUPR_tensors'])
    plt.title('Model micro AUPR')
    aupr_string = param_dict_string + '_aupr.png'
    print('Saving ' + aupr_string)
    plt.savefig(aupr_string)
    
    return history, model 


def temporal_holdout(X_train, y_train, X_valid, y_valid, X_test, y_test, ker='rbf'):
    y_scores = np.zeros((y_train.shape[1]), dtype=np.float)

    print('This should do nothing:')
    X_train, y_train = remove_zero_annot_rows(X_train, y_train)
    X_valid, y_valid = remove_zero_annot_rows(X_valid, y_valid)
    X_test, y_test = remove_zero_annot_rows(X_test, y_test)

    # now I have a training and testing X and y with no zero rows.
    # Make kernels for X_train and X_test
    # range of hyperparameters
    C_range = 10.**np.arange(-1, 3)
    if ker == 'rbf':
        gamma_range = 10.**np.arange(-3, 1)
    elif ker == 'lin':
        gamma_range = [0]
    else:
        print ("### Wrong kernel.")

    # pre-generating kernels
    print ("### Pregenerating kernels...")
    K_rbf_train = {}
    K_rbf_valid = {}
    K_rbf_test = {}
    for gamma in gamma_range:
        K_rbf_train[gamma] = kernel_func(X_train, param=gamma) # K_rbf_train has the same indices as y_train and X_train
        K_rbf_valid[gamma] = kernel_func(X_valid, X_train, param=gamma) # K_rbf_test has the same indices as y_test and X_test on axis 0 and X_train on axis 1
        K_rbf_test[gamma] = kernel_func(X_test, X_train, param=gamma) # K_rbf_test has the same indices as y_test and X_test on axis 0 and X_train on axis 1
    print ("### Done.")

    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    print ("Train samples=%d; Valid samples=%d, #Test samples=%d" % (y_train.shape[0], y_valid.shape[0], y_test.shape[0]))

    # parameter fitting
    C_opt = None
    gamma_opt = None
    max_aupr = 0
    for C in C_range:
        for gamma in gamma_range:
            # Multi-label classification
            cv_results = []
            clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                              random_state=123,
                                              probability=True), n_jobs=-1)
            y_score_valid = np.zeros(y_valid.shape, dtype=float)
            y_pred_valid = np.zeros_like(y_valid)
            print(y_train.shape)
            print(K_rbf_train[gamma].shape)
            clf.fit(K_rbf_train[gamma], y_train)
            y_score_valid = clf.predict_proba(K_rbf_valid[gamma])
            y_pred_valid = clf.predict(K_rbf_valid[gamma])
            perf_th_val = evaluate_performance(y_valid,
                                           y_score_valid,
                                           y_pred_valid)
            curr_micro = perf_th_val['pr_micro']
            print ("### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, cv_aupr))
            if curr_micro > max_aupr:
                C_opt = C
                gamma_opt = gamma
                max_aupr = curr_micro
    print ("### Optimal parameters: ")
    print ("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
    print ("### Train dataset: AUPR = %0.3f" % (max_aupr))
    print
    print ("### Using full training data...")
    clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                      random_state=123,
                                      probability=True), n_jobs=-1)
    y_score = np.zeros(y_test.shape, dtype=float)
    y_pred = np.zeros_like(y_test)
    K_full_train = np.concatenate([K_rbf_train[gamma_opt], K_rbf_valid[gamma_opt]], axis=0)
    y_full_train = np.concatenate([y_train, y_valid], axis=0)
    clf.fit(K_full_train, y_full_train)

    # Compute performance on test set
    y_score = clf.predict_proba(K_rbf_test[gamma_opt])
    y_pred = clf.predict(K_rbf_test[gamma_opt])
    perf = evaluate_performance(y_test, y_score, y_pred)
    for go_id in range(0, y_pred.shape[1]):
        y_scores[go_id] = perf[go_id]
    print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf['pr_micro'], perf['pr_macro'], perf['F1'], perf['acc']))
    print
    print

    return perf, y_scores


def output_projection_files(X, y, model_name, ont, label_names):
    np.savetxt('./features_and_labels/' + model_name + '_' + ont + '_projection_features.tsv', X, delimiter='\t')
    np.savetxt('./features_and_labels/' + model_name + '_' + ont + '_projection_labels.tsv', y, delimiter='\t', header='\t'.join(label_names))
    

def train_and_predict_all_orgs(X, y, X_to_pred, pred_protein_names, go_terms, keyword, ont, arch_set=None):
    """Train on all proteins with annotations, and predicts on all proteins in X_to_pred"""
    # Each row in X and y should correspond with each other
    # X_to_pred should be the network features that align with pred_protein_names
    pred_file = {'prot_IDs': pred_protein_names,
                 'GO_IDs': go_terms,
                 'preds': np.zeros((len(pred_protein_names), len(go_terms))), # fill up this matrix once model is trained and ready to predict
                 }

    # filter samples with no annotations in order to train classifier
    X_with_annot, y_with_annot = remove_zero_annot_rows(X, y)

    print ("Train samples=" + str(y.shape[0]))

    if arch_set == 'bac':
        # for bacteria
        print("RUNNING MODEL ARCHITECTURE FOR BACTERIA")
        params = BAC_PARAMS
    elif arch_set == 'euk':
        # for eukaryotes
        print("RUNNING MODEL ARCHITECTURES FOR EUKARYOTES")
        params = EUK_PARAMS
    else:
        print('No arch_set chosen! Need to specify in order to know which hyperparameter sets to search through for cross-validation using neural networks with original features.')

    params = {param_name:param_list[0] for (param_name, param_list) in params.items()}

    ensure_dir('hyperparam_searches/')
    exp_name = 'hyperparam_searches/' + keyword + '-' + ont
    params['exp_name'] = exp_name
    print ("### Using full training data...")
    history, model = build_and_fit_nn_classifier(X_with_annot, y_with_annot, params, verbose=1)

    # Predict for those proteins in X_to_pred
    pred_file['preds'][:, :] = model.predict(X_to_pred)

    return pred_file


def remove_zero_annot_rows_w_labels(X, y, protein_names):
    print("before remove_zero_annot_rows")
    print(X.shape)
    X, _ = remove_zero_annot_rows(X, y)
    protein_names, y = remove_zero_annot_rows(np.array(protein_names), y)
    return X, y, protein_names


def one_spec_cross_val(X_test_species, y_test_species, test_species_prots, 
        X_rest, y_rest, rest_prot_names, go_terms, keyword, ont, 
        n_trials=5, num_hyperparam_sets=25, arch_set=None, save_only=False, load_file=None, subsample=False):
    """Perform model selection via 5-fold cross validation"""
    # if supplied load_file, need to get X_test_species, y_test_species, test_species_prots, X_rest, y_rest, rest_prot_names, go_terms
    # (basically all data arguments) from the load_file.
    if load_file is None:
        # filter samples with no annotations
        X_test_species, y_test_species, test_species_prots = remove_zero_annot_rows_w_labels(X_test_species, 
                y_test_species, test_species_prots)
        X_rest, y_rest, rest_prots = remove_zero_annot_rows_w_labels(X_rest, y_rest, 
                rest_prot_names)
        trial_file = {}
        trial_file['X_rest'] = X_rest
        trial_file['Y_rest'] = y_rest
        trial_file['rest_prot_names'] = rest_prot_names
        trial_file['test_goids'] = go_terms
        trial_file['X_test_species'] = X_test_species
        trial_file['Y_test_species'] = y_test_species
        trial_file['test_species_prots'] = test_species_prots
        pickle.dump(trial_file, open('./train_test_data/' + keyword + '_' + ont + '_one_spec_train_test_data_file.pckl', 'wb'), protocol=4)
        if save_only:
            exit()
    else:
        trial_file = pickle.load(open('./train_test_data/' + keyword + '_' + ont + '_one_spec_train_test_data_file.pckl', 'wb'))
    X_rest = trial_file['X_rest']
    y_rest = trial_file['Y_rest']
    rest_prot_names = trial_file['rest_prot_names']
    go_terms = trial_file['test_goids']
    X_test_species = trial_file['X_test_species']
    y_test_species = trial_file['Y_test_species']
    test_species_prots = trial_file['test_species_prots']
    
    print("Shapes of X and Y matrices")
    print('X_test_species')
    print(X_test_species.shape)
    print('X_rest')
    print(X_rest.shape)
    print('y_test_species')
    print(y_test_species.shape)
    print('y_rest')
    print(y_rest.shape)
    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    # shuffle and split training and test sets
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=1)
    ss = trials.split(X_test_species)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))


    y_score_trials = np.zeros((y_test_species.shape[1], n_trials), dtype=np.float)
    it = 0
    pred_file = {'prot_IDs': test_species_prots,
                 'GO_IDs': go_terms,
                 'trial_preds': np.zeros((n_trials, len(test_species_prots), len(go_terms))),
                 'trial_splits': trial_splits,
                 'true_labels': y_test_species,
                 'trial_hyperparams': []
                 }
    for jj in range(0, n_trials):
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
        it += 1
        X_test_species_train, X_test_species_val, y_test_species_train, y_test_species_val = train_test_split(
                X_test_species[train_idx], y_test_species[train_idx], test_size=0.1) # create validation set from species to be tested on
        X_train = np.concatenate((X_test_species_train, X_rest), axis=0)
        X_test = X_test_species[test_idx]
        y_train = np.concatenate((y_test_species_train, y_rest), axis=0)
        y_test = y_test_species[test_idx]
        if subsample:
            num_to_subsample = X_test_species_train.shape[0] # this is the number of elements to get
            inds_to_subsample = np.arange(X_train.shape[0])
            subsample_inds = np.random.choice(inds_to_subsample, size=num_to_subsample, replace=False)
            print('Subsampling! Number of subsample inds: ' + str(num_to_subsample))
            X_train = X_train[subsample_inds, :]
            y_train = y_train[subsample_inds, :]
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print ("### [Trial %d] Perfom cross validation...." % (it))
        print ("Train samples=%d; #Validation samples=%d #Test samples=%d" % (y_train.shape[0], y_test_species_val.shape[0], y_test.shape[0]))

        ensure_dir('hyperparam_searches/')
        exp_name = 'hyperparam_searches/' + keyword + '-' + ont + '-fold-' + str(jj)
        if arch_set == 'bac':
            # for bacteria
            print("RUNNING MODEL ARCHITECTURE FOR BACTERIA")
            params = BAC_PARAMS
            if int(num_hyperparam_sets) == 1:
                print('Not searching through params')
                params = BAC_PARAMS_NO_SEARCH
        elif arch_set == 'euk':
            # for eukaryotes
            print("RUNNING MODEL ARCHITECTURES FOR EUKARYOTES")
            params = EUK_PARAMS
            if int(num_hyperparam_sets) == 1:
                print('Not searching through params')
                params = EUK_PARAMS_NO_SEARCH
        else:
            print('No arch_set chosen! Need to specify in order to know which hyperparameter sets to search through for cross-validation using neural networks with original features.')

        exp_path = exp_name + '_num_hyperparam_sets_' + str(num_hyperparam_sets)
        print(type(num_hyperparam_sets))
        if num_hyperparam_sets > 1:
            # hyperparam search
            print('number of hyperparam sets to train for this trial:' + str(num_hyperparam_sets))
            params['in_shape'] = [X_train.shape[1]]
            params['out_shape'] = [y_train.shape[1]]
            keras_model = KerasClassifier(build_fn=build_maxout_nn_classifier)
            clf = RandomizedSearchCV(keras_model, params, cv=2, n_iter=num_hyperparam_sets, scoring=make_scorer(micro_AUPR, greater_is_better=True)) # this is training on half the training data, should probably not do this
            search_result = clf.fit(X_train, y_train)
            # summarize results
            means = search_result.cv_results_['mean_test_score']
            stds = search_result.cv_results_['std_test_score']
            params = search_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
            print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
            best_params = search_result.best_params_
            print('Best model parameters for this trial:')
            print(best_params)
            print ("### Using full training data...")
            with open(exp_path + '_search_results.pckl', 'wb') as search_result_file:
                pickle.dump(search_result, search_result_file)
            del best_params['in_shape']
            del best_params['out_shape']
        else:
            # no hyperparam search
            best_params = {param_name:param_list[0] for (param_name, param_list) in params.items()}
        best_params['exp_name'] = exp_name
        history, model = build_and_fit_nn_classifier(X_train, y_train, best_params, X_val=X_test_species_val, y_val=y_test_species_val, verbose=1)

        y_score = np.zeros(y_test.shape, dtype=float)
        y_pred = np.zeros_like(y_test)

        # Compute performance on test set
        y_score = model.predict(X_test_species[test_idx])
        pred_file['trial_preds'][jj, :, :] = model.predict(X_test_species)
        pred_file['trial_hyperparams'].append(best_params)
        y_pred = y_score > 0.5 #silly way to do predictions from the scores; choose threshold, maybe use platt scaling or something else
        perf_trial = evaluate_performance(y_test, y_score, y_pred)
        for go_id in range(0, y_pred.shape[1]):
            y_score_trials[go_id, jj] = perf_trial[go_id]
        pr_micro.append(perf_trial['pr_micro'])
        pr_macro.append(perf_trial['pr_macro'])
        F1.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['pr_micro'], perf_trial['pr_macro'], perf_trial['F1'], perf_trial['acc']))
        print
        print


    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['F1'] = F1
    perf['acc'] = acc

    return perf, y_score_trials, pred_file
     


def cross_validation_nn(X, y, protein_names, go_terms, keyword, ont, n_trials=5, num_hyperparam_sets=25, arch_set=None, save_only=False, load_file=None):
    """Perform model selection via 5-fold cross validation"""
    if load_file is None:
        # filter samples with no annotations
        print("before remove_zero_annot_rows")
        print(X.shape)
        X, _ = remove_zero_annot_rows(X, y)
        protein_names, y = remove_zero_annot_rows(np.array(protein_names), y)

        if save_only:
            data_file = {}
            data_file['X'] = X
            data_file['Y'] = y
            data_file['prot_names'] = protein_names
            data_file['test_goids'] = go_terms
            dump_fname = './train_test_data/' + keyword + '_' + ont + '_cross_validation_train_test_data_file.pckl'
            pickle.dump(data_file, open(dump_fname, 'wb'), protocol=4)
            exit()
    else:
        data_file = pickle.load(open(load_file, 'rb'))
        X = data_file['X']
        y = data_file['Y']
        protein_names = data_file['prot_names']
        go_terms = data_file['test_goids']

    # performance measures
    pr_micro = []
    pr_macro = []
    F1 = []
    acc = []

    # shuffle and split training and test sets
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=1)
    ss = trials.split(X)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))

    y_score_trials = np.zeros((y.shape[1], n_trials), dtype=np.float)
    it = 0
    pred_file = {'prot_IDs': protein_names,
                 'GO_IDs': go_terms,
                 'trial_preds': np.zeros((n_trials, len(protein_names), len(go_terms))),
                 'trial_splits': trial_splits,
                 'true_labels': y
                 }
    for jj in range(0, n_trials):
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
        it += 1
        X_train = X[train_idx] 
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        print ("### [Trial %d] Perfom cross validation...." % (it))
        print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))
        #downsample_rate = 0.01 # for bacteria
        #downsample_rate = 0.001 # for eukaryotes
        ensure_dir('hyperparam_searches/')
        exp_name = 'hyperparam_searches/' + keyword + '-' + ont + '-fold-' + str(jj)
        if arch_set == 'bac':
            # for bacteria
            print("RUNNING MODEL ARCHITECTURE FOR BACTERIA")
            params = BAC_PARAMS
        elif arch_set == 'euk':
            # for eukaryotes
            print("RUNNING MODEL ARCHITECTURES FOR EUKARYOTES")
            params = EUK_PARAMS
        else:
            print('No arch_set chosen! Need to specify in order to know which hyperparameter sets to search through for cross-validation using neural networks with original features.')

        exp_path = exp_name + '_num_hyperparam_sets_' + str(num_hyperparam_sets)
        # hyperparam search
        if num_hyperparam_sets > 1:
            print('number of hyperparam sets to train for this trial:' + str(num_hyperparam_sets))
            params['in_shape'] = [X_train.shape[1]]
            params['out_shape'] = [y_train.shape[1]]
            keras_model = KerasClassifier(build_fn=build_maxout_nn_classifier)
            clf = RandomizedSearchCV(keras_model, params, cv=2, n_iter=num_hyperparam_sets, scoring=make_scorer(micro_AUPR, greater_is_better=True))
            search_result = clf.fit(X_train, y_train)
            # summarize results
            means = search_result.cv_results_['mean_test_score']
            stds = search_result.cv_results_['std_test_score']
            params = search_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                    print("%f (%f) with: %r" % (mean, stdev, param))
            print("Best: %f using %s" % (search_result.best_score_, search_result.best_params_))
            best_params = search_result.best_params_
            print('Best model parameters for this trial:')
            print(best_params)
            print ("### Using full training data...")
            with open(exp_path + '_search_results.pckl', 'wb') as search_result_file:
                pickle.dump(search_result, search_result_file)
            del best_params['in_shape']
            del best_params['out_shape']
        else:
            # no hyperparam search
            best_params = {param_name:param_list[0] for (param_name, param_list) in params.items()}
        best_params['exp_name'] = exp_name
        history, model = build_and_fit_nn_classifier(X[train_idx, :], y_train, best_params, verbose=1)

        y_score = np.zeros(y_test.shape, dtype=float)
        y_pred = np.zeros_like(y_test)

        # Compute performance on test set
        y_score = model.predict(X_test)
        pred_file['trial_preds'][jj, :, :] = model.predict(X)
        y_pred = y_score > 0.5 #silly way to do predictions from the scores; choose threshold, maybe use platt scaling or something else
        perf_trial = evaluate_performance(y_test, y_score, y_pred)
        for go_id in range(0, y_pred.shape[1]):
            y_score_trials[go_id, jj] = perf_trial[go_id]
        pr_micro.append(perf_trial['pr_micro'])
        pr_macro.append(perf_trial['pr_macro'])
        F1.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f\n\n" % (perf_trial['pr_micro'], perf_trial['pr_macro'], perf_trial['F1'], perf_trial['acc']))

    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['F1'] = F1
    perf['acc'] = acc

    return perf, y_score_trials, pred_file

