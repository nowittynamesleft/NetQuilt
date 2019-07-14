import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, cosine_similarity
from sklearn.utils import resample
from keras.models import Model
from keras.layers import Input, Dense, maximum, BatchNormalization, Dropout
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
import talos as ta
from talos.metrics.keras_metrics import fmeasure_acc
from talos.utils.gpu_utils import multi_gpu
import matplotlib.pyplot as plt


def kernel_func(X, Y=None, param=0):
    if param != 0:
        K = rbf_kernel(X, Y, gamma=param)
    else:
        K = linear_kernel(X, Y)

    return K


def real_AUPR(label, score):
    """Computing real AUPR . By Vlad and Meet"""
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
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0

    return pr, f


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
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    perf["M-aupr"] /= n

    # Compute micro-averaged AUPR
    perf["m-aupr"], _ = real_AUPR(y_test, y_score)

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


def temporal_holdout(X, y, indx, goterms, bootstraps=None, ker='rbf'):
    """Perform temporal holdout validation"""
    X_train = X[indx['train'].tolist()]
    X_test = X[indx['test'].tolist()]
    X_valid = X[indx['valid'].tolist()]
    y_train = np.array(y['train'].tolist())
    y_test = np.array(y['test'].tolist())
    y_valid = np.array(y['valid'].tolist())
    goterms = goterms['terms'].tolist()

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
    K_rbf_valid = {}
    for gamma in gamma_range:
        K_rbf_train[gamma] = kernel_func(X_train, param=gamma)
        K_rbf_test[gamma] = kernel_func(X_test, X_train, param=gamma)
        K_rbf_valid[gamma] = kernel_func(X_valid, X_train, param=gamma)
    print ("### Done.")
    print ("Train samples=%d; #Test samples=%d" % (y_train.shape[0], y_test.shape[0]))

    # parameter fitting
    C_opt = None
    gamma_opt = None
    max_aupr = 0
    for C in C_range:
        for gamma in gamma_range:
            # Multi-label classification
            clf = OneVsRestClassifier(svm.SVC(C=C, kernel='precomputed',
                                              random_state=123,
                                              probability=True), n_jobs=-1)
            clf.fit(K_rbf_train[gamma], y_train)
            # y_score_valid = clf.decision_function(K_rbf_valid[gamma])
            y_score_valid = clf.predict_proba(K_rbf_valid[gamma])
            y_pred_valid = clf.predict(K_rbf_valid[gamma])
            perf = evaluate_performance(y_valid,
                                        y_score_valid,
                                        y_pred_valid)
            micro_aupr = perf['m-aupr']
            print ("### gamma = %0.3f, C = %0.3f, AUPR = %0.3f" % (gamma, C, micro_aupr))
            if micro_aupr > max_aupr:
                C_opt = C
                gamma_opt = gamma
                max_aupr = micro_aupr
    print ("### Optimal parameters: ")
    print ("C_opt = %0.3f, gamma_opt = %0.3f" % (C_opt, gamma_opt))
    print ("### Train dataset: AUPR = %0.3f" % (max_aupr))
    print
    print ("### Computing performance on test dataset...")
    clf = OneVsRestClassifier(svm.SVC(C=C_opt, kernel='precomputed',
                                      random_state=123,
                                      probability=True), n_jobs=-1)
    clf.fit(K_rbf_train[gamma_opt], y_train)

    # Compute performance on test set
    # y_score = clf.decision_function(K_rbf_test[gamma_opt])
    y_score = clf.predict_proba(K_rbf_test[gamma_opt])
    y_pred = clf.predict(K_rbf_test[gamma_opt])

    # performance measures for bootstrapping
    pr_micro = []
    pr_macro = []
    fmax = []
    acc = []

    # individual goterms
    pr_goterms = {}
    for i in range(0, len(goterms)):
        pr_goterms[goterms[i]] = []

    # bootstraps
    if bootstraps is None:
        # generate indices for bootstraps
        bootstraps = []
        for i in range(0, 1000):
            bootstraps.append(resample(np.arange(y_test.shape[0])))
    else:
        pass

    for ind in bootstraps:
        perf_ind = evaluate_performance(y_test[ind],
                                        y_score[ind],
                                        y_pred[ind])
        pr_micro.append(perf_ind['m-aupr'])
        pr_macro.append(perf_ind['M-aupr'])
        fmax.append(perf_ind['F1'])
        acc.append(perf_ind['acc'])
        for i in range(0, len(goterms)):
            pr_goterms[goterms[i]].append(perf_ind[i])

    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['fmax'] = fmax
    perf['acc'] = acc
    perf['pr_goterms'] = pr_goterms

    return perf

def train_test(X, y, train_idx, test_idx, ker='rbf'):

    X_train = X[train_idx, :]
    X_test = X[test_idx, :]
    y_scores = np.zeros((y.shape[1], 1), dtype=np.float)
    y_train = y[train_idx, :]
    y_test = y[test_idx, :]
    print('Y_train before')
    print(y_train.shape)
    print('Y_test before')
    print(y_test.shape)
    print('X_train before')
    print(X_train.shape)
    print('X_test before')
    print(X_test.shape)

    print('This should do nothing (unless there are 0 rows):')
    del_rid = np.where(y_train.sum(axis=1) == 0)[0]
    y_train = np.delete(y_train, del_rid, axis=0)
    X_train = np.delete(X_train, del_rid, axis=0)

    del_rid = np.where(y_test.sum(axis=1) == 0)[0]
    y_test = np.delete(y_test, del_rid, axis=0)
    X_test = np.delete(X_test, del_rid, axis=0)
    print('Y_train after')
    print(y_train.shape)
    print('Y_test after')
    print(y_test.shape)
    print('X_train after')
    print(X_train.shape)
    print('X_test after')
    print(X_test.shape)

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
        print('Generated kernel shape')
        K_rbf_train[gamma] = kernel_func(X_train, param=gamma) # K_rbf_train has the same indices as y_train and X_train
        print(K_rbf_train[gamma].shape)
        assert K_rbf_train[gamma].shape == (y_train.shape[0], y_train.shape[0])
        K_rbf_test[gamma] = kernel_func(X_test, X_train, param=gamma) # K_rbf_test has the same indices as y_test and X_test on axis 0 and X_train on axis 1
        print(K_rbf_test[gamma].shape)
        assert K_rbf_test[gamma].shape == (y_test.shape[0], y_train.shape[0])

    print ("### Done.")

    # performance measures
    pr_micro = []
    pr_macro = []
    fmax = []
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
                print('Inner kernel shape')
                K_train = K_rbf_train[gamma][train, :][:, train]
                print(K_train.shape)
                K_valid = K_rbf_train[gamma][valid, :][:, train]
                print(K_valid.shape)
                y_train_t = y_train[train]
                y_train_v = y_train[valid]
                y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                y_pred_valid = np.zeros_like(y_train_v)
                idx = np.where(y_train_t.sum(axis=0) > 0)[0] # why is this necessary? oh, not all go terms definitely have labels for random splits. gotcha
                clf.fit(K_train, y_train_t[:, idx])
                y_score_valid[:, idx] = clf.predict_proba(K_valid)
                y_pred_valid[:, idx] = clf.predict(K_valid)
                perf_cv = evaluate_performance(y_train_v,
                                               y_score_valid,
                                               y_pred_valid)
                cv_results.append(perf_cv['m-aupr'])
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
    y_score = clf.predict_proba(K_rbf_test[gamma_opt])
    y_pred = clf.predict(K_rbf_test[gamma_opt])
    perf_trial = evaluate_performance(y_test, y_score, y_pred)
    for go_id in range(0, y_pred.shape[1]):
        y_scores[go_id] = perf_trial[go_id]
    pr_micro = perf_trial['m-aupr']
    pr_macro = perf_trial['M-aupr']
    fmax = perf_trial['F1']
    acc = perf_trial['acc']
    print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['m-aupr'], perf_trial['M-aupr'], perf_trial['F1'], perf_trial['acc']))
    print
    print

    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['fmax'] = fmax
    perf['acc'] = acc

    return perf, y_scores


def cross_validation(X, y, n_trials=5, ker='rbf', X_pred=None):
    """Perform model selection via 5-fold cross validation"""
    # filter samples with no annotations
    del_rid = np.where(y.sum(axis=1) == 0)[0]
    y = np.delete(y, del_rid, axis=0)
    X = np.delete(X, del_rid, axis=0)
    print('X and y shape:')
    print(X)
    print(y)
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
    fmax = []
    acc = []

    # shuffle and split training and test sets
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
    ss = trials.split(X)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))

    y_score_trials = np.zeros((y.shape[1], n_trials), dtype=np.float)
    it = 0
    for jj in range(0, n_trials):
        train_idx = trial_splits[jj][0]
        test_idx = trial_splits[jj][1]
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
                    cv_results.append(perf_cv['m-aupr'])
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
            y_score_trials[go_id, jj] = perf_trial[go_id]
        pr_micro.append(perf_trial['m-aupr'])
        pr_macro.append(perf_trial['M-aupr'])
        fmax.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['m-aupr'], perf_trial['M-aupr'], perf_trial['F1'], perf_trial['acc']))
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
    perf['fmax'] = fmax
    perf['acc'] = acc

    return perf, y_score_trials, y_score_pred


def build_nn_classifier(input_dim, output_dim, hidden_dim):
    input_layer = Input(shape=(input_dim,))
    hidden_layer = Dense(hidden_dim, activation='sigmoid')(input_layer)
    output_layer = Dense(output_dim, activation='sigmoid')(hidden_layer)
    model = Model(input=input_layer, output=output_layer)

    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print(model.summary())
    
    return model 


def build_maxout_nn_classifier(input_dim, output_dim, maxout_units): # this is all the hyperparameters of Wan et al.
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
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[fmeasure_acc])
    model.summary()
    return model


def build_and_fit_nn_classifier(X_train, y_train, X_val, y_val, params):
    input_layer = Input(shape=(X_train.shape[1],))
    '''
    hidden_dim_1 = int(params['hidden_dim_1'])
    hidden_dim_2 = int(params['hidden_dim_2']) if params['hidden_dim_2'] != 'None' else None
    hidden_dim_3 = int(params['hidden_dim_3']) if params['hidden_dim_3'] != 'None' else None
    hidden_dim_4 = int(params['hidden_dim_4']) if params['hidden_dim_4'] != 'None' else None
    '''
    hidden_dim_1 = params['hidden_dim_1']
    hidden_dim_2 = params['hidden_dim_2']
    hidden_dim_3 = params['hidden_dim_3']
    hidden_dim_4 = params['hidden_dim_4']
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
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[fmeasure_acc])
    model.summary()
    
    history = model.fit(X_train, y_train, validation_data=[X_val, y_val], batch_size=int(params['batch_size']),  epochs=int(params['num_epochs']), verbose=0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('classifier_loss_plots/' + str(params['exp_name']) + '.png')
    
    return history, model 


def cross_validation_nn(X, y, n_trials=5, X_pred=None):
    """Perform model selection via 5-fold cross validation"""
    NUM_EPOCHS = 150
    # filter samples with no annotations
    del_rid = np.where(y.sum(axis=1) == 0)[0]
    y = np.delete(y, del_rid, axis=0)
    X = np.delete(X, del_rid, axis=0)
    hidden_dims = [3]

    if X_pred is not None:
        y_score_pred = np.zeros((X_pred.shape[0], y.shape[1]), dtype=np.float)

    # performance measures
    pr_micro = []
    pr_macro = []
    fmax = []
    acc = []

    # shuffle and split training and test sets
    trials = ShuffleSplit(n_splits=n_trials, test_size=0.2, random_state=None)
    ss = trials.split(X)
    trial_splits = []
    for train_idx, test_idx in ss:
        trial_splits.append((train_idx, test_idx))

    y_score_trials = np.zeros((y.shape[1], n_trials), dtype=np.float)
    it = 0
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
        '''
        # setup for nested cross-validation
        splits = ml_split(y_train)

        # parameter fitting
        max_aupr = -1
        for hidden_dim in hidden_dims:
            cv_results = []
            for train, valid in splits:
                # Define current nested split training/validation data
                X_train_t = X_train[train]
                X_train_v = X_train[valid]

                y_train_t = y_train[train]
                y_train_v = y_train[valid]
                y_score_valid = np.zeros(y_train_v.shape, dtype=float)
                y_pred_valid = np.zeros_like(y_train_v)
                idx = np.where(y_train_t.sum(axis=0) > 0)[0] # idx is list of go term inds for which there are any training examples


                model = build_maxout_nn_classifier(X_train.shape[1], y_train.shape[1], hidden_dim)
                model.fit(X_train_t, y_train_t[:, idx], epochs=NUM_EPOCHS)
                y_score_valid[:, idx] = model.predict(X_train_v)
                y_pred_valid[:, idx] = y_score_valid[:, idx] > 0.5 #silly way to do predictions from the scores; choose threshold, maybe use platt scaling or something else

                perf_cv = evaluate_performance(y_train_v,
                                               y_score_valid,
                                               y_pred_valid)
                cv_results.append(perf_cv['m-aupr'])
            cv_aupr = np.median(cv_results)
            print ("### hidden_dim = %d, AUPR = %0.3f" % (hidden_dim, cv_aupr))
            if cv_aupr > max_aupr:
                hidden_dim_opt = hidden_dim
                max_aupr = cv_aupr 
        print ("### Optimal parameters: ")
        print ("hidden_dim = %d" % (hidden_dim_opt))
        print ("### Train dataset: AUPR = %0.3f" % (max_aupr))
        print
        print ("### Using full training data...")
        '''
        #downsample_rate = 0.01
        #exp_name = 'first_' + str(downsample_rate) + '_sampled_model_orgs_all_annots_human_goids_hyperparam_search'
        #downsample_rate = 0.01
        #exp_name = 'orig_features_test'
        downsample_rate = 0.001
        exp_name = 'orig_features_model_orgs_tiny_test'
        params = {'hidden_dim_1': [500, 1000],
                    'hidden_dim_2': [200, 700, 0],
                    'hidden_dim_3': [300, 800],
                    'hidden_dim_4': [800, 1000, 1500],
                    'maxout_units': [3, 4, 5],
                    'dropout': [0.2, 0.3, 0.4, 0.5],
                    'num_epochs': [200, 250, 300],
                    'learning_rate': [0.01],
                    'activation': ['relu'],
                    'batch_size': [16, 32],
                    'exp_name': [exp_name]
        }
        #exp_name = 'third_' + str(downsample_rate) + '_sampled_human_hyperparam_search'
        exp = str(jj)
        results = ta.Scan(X_train, y_train, model=build_and_fit_nn_classifier, params=params, val_split=0.2, print_params=True, grid_downsample=downsample_rate, dataset_name=exp_name, experiment_no=exp)
        #report = ta.Reporting(exp_name + '_' + exp + '.csv')
        print ("### Using full training data...")
        best_p = results.data.sort_values('val_fmeasure_acc', ascending=False).to_dict('records')[0] # weirdly, there's no best parameters function in talos for the results that gets a dictionary that was the same as the input to the model
        best_p['maxout_units'] = int(best_p['maxout_units'])
        best_p['hidden_dim_1'] = int(best_p['hidden_dim_1'])
        best_p['hidden_dim_2'] = int(best_p['hidden_dim_2'])
        best_p['hidden_dim_3'] = int(best_p['hidden_dim_3'])
        best_p['hidden_dim_4'] = int(best_p['hidden_dim_4'])
        print('Best model parameters for this trial:')
        print(best_p)
        history, model = build_and_fit_nn_classifier(X[train_idx, :], y_train, X[train_idx, :], y_train, params=best_p)

        #model = build_maxout_nn_classifier(X_train.shape[1], y_train.shape[1], hidden_dim_opt)
        #model.fit(X[train_idx, :], y_train, epochs=NUM_EPOCHS)
        y_score = np.zeros(y_test.shape, dtype=float)
        y_pred = np.zeros_like(y_test)

        # Compute performance on test set
        y_score = model.predict(X[test_idx])
        y_pred = y_score > 0.5 #silly way to do predictions from the scores; choose threshold, maybe use platt scaling or something else
        perf_trial = evaluate_performance(y_test, y_score, y_pred)
        for go_id in range(0, y_pred.shape[1]):
            y_score_trials[go_id, jj] = perf_trial[go_id]
        pr_micro.append(perf_trial['m-aupr'])
        pr_macro.append(perf_trial['M-aupr'])
        fmax.append(perf_trial['F1'])
        acc.append(perf_trial['acc'])
        print ("### Test dataset: AUPR['micro'] = %0.3f, AUPR['macro'] = %0.3f, F1 = %0.3f, Acc = %0.3f" % (perf_trial['m-aupr'], perf_trial['M-aupr'], perf_trial['F1'], perf_trial['acc']))
        print
        print
        if X_pred is not None:
            print ("### Predicting functions...")
            y_score_pred += model.predict(X_pred)


    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['fmax'] = fmax
    perf['acc'] = acc
    y_score_pred /= n_trials

    return perf, y_score_trials, y_score_pred

