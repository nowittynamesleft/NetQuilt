import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD, Adam
from keras import regularizers
import numpy as np
import time
from keras.losses import categorical_crossentropy
from sklearn.metrics.pairwise import cosine_similarity


'''
def sigmoid_sim_loss(S, F):
    F_2 = tf.matmul(F, tf.transpose(F)) # 64x64
    sigmoid = tf.sigmoid(F_2)
    epsilon = 1e-6 # epsilon to prevent nans
    grand_sum = -tf.reduce_sum(S*tf.log(sigmoid + epsilon) + (1-S)*tf.log(1 - sigmoid + epsilon))

    return tf.divide(grand_sum, tf.cast(tf.square(tf.shape(S)[0]), tf.float32))
'''


def sigmoid_sim_loss_label_mask(M_S, F):
    F_2 = tf.matmul(F, tf.transpose(F)) # 64x64
    # first column of M_S is always whether the sample i is labeled or unlabeled
    M = M_S[:, 0:1]
    S = M_S[:, 1:]
    mask = tf.matmul(M, tf.transpose(M)) # if S is nx1, M is n x n, with 0 columns and 0 rows for unlabeled proteins

    S_masked = tf.boolean_mask(S, mask)
    F_2_masked = tf.boolean_mask(F_2, mask)

    sigmoid_masked = tf.sigmoid(F_2_masked)
    epsilon = 1e-6 # epsilon to prevent nans
    grand_sum = -tf.reduce_sum(S_masked*tf.log(sigmoid_masked + epsilon) + (1-S_masked)*tf.log(1 - sigmoid_masked + epsilon))
    loss = tf.cond(tf.equal(tf.shape(S_masked)[0], 0), lambda: 0.0, lambda: tf.divide(grand_sum, tf.reduce_sum(tf.cast(mask, tf.float32)))) # if the shape of the similarity mat is 0, return 0, otherwise return the grand_sum
    # divide by the number of chosen interactions
    return loss


def real_graph_laplacian_loss(L, batch_feat_vecs):
    product = tf.matmul(tf.matmul(tf.transpose(batch_feat_vecs), L), batch_feat_vecs)
    #return tf.trace(product) # before
    print(tf.to_float(tf.shape(L)[0]))
    return tf.trace(product)/tf.to_float(tf.shape(L)[0]) # trying to account for the batch size affecting the loss


def sigmoid_sim_loss_subsample(S, F):
    F_2 = tf.matmul(F, tf.transpose(F)) # 64x64

    pos_mask = tf.equal(S, 1)
    total_mask = tf.logical_or(tf.greater_equal(tf.random_uniform(tf.shape(S)), 0.5), pos_mask) # sample half of the negatives

    S_masked = tf.boolean_mask(S, total_mask)
    F_2_masked = tf.boolean_mask(S, total_mask)

    sigmoid_masked = tf.sigmoid(F_2_masked)
    epsilon = 1e-6 # epsilon to prevent nans
    grand_sum = -tf.reduce_sum(S_masked*tf.log(sigmoid_masked + epsilon) + (1-S_masked)*tf.log(1 - sigmoid_masked + epsilon))

    return tf.divide(grand_sum, tf.reduce_sum(tf.cast(total_mask, tf.float32))) # divide by the number of chosen interactions


def build_sim_reg_AE(input_dim, encoding_dims, sim_reg_lamb, hidden_activation='tanh'):
    """
    Function for building autoencoder.
    """
    # input layer
    input_layer = Input(shape=(input_dim, ))
    hidden_layer = input_layer
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='middle_layer')(hidden_layer)
            mid_layer = hidden_layer
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,
                    activation='sigmoid')(hidden_layer)

    output_layers = [mid_layer, decoded]
    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=output_layers)

    losses = [sigmoid_sim_loss_label_mask, 'binary_crossentropy']
    loss_weights = [sim_reg_lamb, 1.]
    model.compile(optimizer=sgd, loss=losses, loss_weights=loss_weights)
    print (model.summary())

    return model


def train_sim_reg_model_given_batches(model, train_epoch_in_list, train_epoch_out_list, val_epoch_in_list, 
                                         val_epoch_out_list, sim_reg_lamb):
    print('training model')
    print('batch size:')
    print(train_epoch_in_list[0][0].shape[0])
    print('Sim reg lamb:')
    print(sim_reg_lamb)
    recon_loss_history = []
    recon_val_loss_history = []
    sim_loss_history = []
    sim_val_loss_history = []
    num_train_batches = len(train_epoch_in_list[0])
    print('Number of train batches: ' + str(num_train_batches))
    num_val_batches = len(val_epoch_in_list[0])
    print('Number of val batches: ' + str(num_val_batches))
    num_epochs = len(train_epoch_in_list)
    for epoch in range(0, num_epochs):
        train_batch_samples_in_list = train_epoch_in_list[epoch]
        train_batch_samples_out_list = train_epoch_out_list[epoch]
        val_batch_samples_in_list = val_epoch_in_list[epoch]
        val_batch_samples_out_list = val_epoch_out_list[epoch]
        time_start = time.time()
        losses = np.array([0., 0.])
        for i in range(0, len(train_batch_samples_in_list)):
            curr_losses = model.train_on_batch(train_batch_samples_in_list[i], train_batch_samples_out_list[i])
            losses[0] += curr_losses[1]
            losses[1] += curr_losses[2]
        val_losses = np.array([0., 0.])
        list_of_losses = []
        for i in range(0, len(val_batch_samples_in_list)):
            curr_losses = model.test_on_batch(val_batch_samples_in_list[i], val_batch_samples_out_list[i])
            val_losses[0] += curr_losses[1] # curr_losses[1] is first individual loss, curr_losses[0] is total los
            val_losses[1] += curr_losses[2] # curr_losses[2] is the second individual loss
            list_of_losses.append(curr_losses)
        effective_sim_val_loss = val_losses[0]*sim_reg_lamb/num_val_batches
        effective_sim_loss = losses[0]*sim_reg_lamb/num_train_batches
        recon_val_loss = val_losses[1]/num_val_batches
        recon_loss = losses[1]/num_train_batches
        av_train_loss = effective_sim_loss + recon_loss
        av_val_loss = effective_sim_val_loss + recon_val_loss
        time_end = time.time()
        duration = time_end - time_start
        print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + ': ')
        print('Average losses: weighted sim train loss for epoch: ' 
            + str(effective_sim_loss) + ' ' + 3*'-' + ' recon train loss for epoch: ' 
            + str(recon_loss) + ' ' + 3*'-' + ' weighted sim val loss for epoch: ' + str(effective_sim_val_loss) + ' '
            + 3*'-' + ' recon val loss for epoch: ' + str(recon_val_loss)
            + ' OVERALL TRAIN LOSS: ' + str(av_train_loss)
            + ' OVERALL VAL LOSS: ' + str(av_val_loss) + ' ' + 3*'-' + ' Elapsed: '  + str(duration) + 's')
        sim_loss_history.append(effective_sim_loss)
        sim_val_loss_history.append(effective_sim_val_loss)
        recon_loss_history.append(recon_loss)
        recon_val_loss_history.append(recon_val_loss)
    return sim_loss_history, sim_val_loss_history, recon_loss_history, recon_val_loss_history


def create_sim_reg_batches(lap_mat_train, lap_mat_valid, X_train_noisy, X_valid_noisy, X_train, X_valid, batch_size=32, num_epochs=5):
    # split X, Y and lap_mat into batches
    inds_train = np.arange(X_train.shape[0])
    inds_test = np.arange(X_valid.shape[0])
    print(inds_train)
    train_epoch_in_list = [] # this will be a list of size num_epochs of lists of size number of batches for training
    train_epoch_out_list = []
    val_epoch_in_list = []
    val_epoch_out_list = []
    for epoch in range(0, num_epochs):
        np.random.shuffle(inds_train)
        batches = np.array_split(inds_train, (len(inds_train)/batch_size) + 1)
        print('Number of batches: ' + str(len(batches)))
        # generate batches before training
        print('Generating batches...')
        batch_time_start = time.time()
        train_batch_samples_in_list = []
        train_batch_samples_out_list = []
        print(lap_mat_train.shape)
        for batch in batches:
            batch_lap_mat = lap_mat_train[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_samples_in = X_train_noisy[batch, :]
            batch_samples_out = [X_train[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            train_batch_samples_in_list.append(batch_samples_in)
            train_batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating batch for epoch ' + str(epoch) + '/' + str(num_epochs) + '. Time it took: ' + str(batch_time))
        val_batch_samples_in_list = []
        val_batch_samples_out_list = []
        print('Generating val batches...')
        batch_time_start = time.time()
        val_batches = np.array_split(inds_test, (len(inds_test)/batch_size) + 1)
        for batch in val_batches:
            batch_lap_mat = lap_mat_valid[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_samples_in = X_valid_noisy[batch, :]
            batch_samples_out = [X_valid[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            val_batch_samples_in_list.append(batch_samples_in)
            val_batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating val batches for epoch ' + str(epoch) + '/' + str(num_epochs) + '. Time it took: ' + str(batch_time))
        train_epoch_in_list.append(train_batch_samples_in_list)
        train_epoch_out_list.append(train_batch_samples_out_list)
        val_epoch_in_list.append(val_batch_samples_in_list)
        val_epoch_out_list.append(val_batch_samples_out_list)

    return train_epoch_in_list, train_epoch_out_list, val_epoch_in_list, val_epoch_out_list


def get_sim_mats(y, train_inds):
    # calculating pairwise equality of binary labels, converting each row of binary digits to decimal in order to calculate equality
    # only get the train similarity matrix for now
    '''
    y has some unannotated rows
    We don't care about that, just compute similarities, but keep track of the unannotated rows in a vector, and return it
    '''
    y_train_with_zero_rows = np.zeros_like(y)
    y_train_with_zero_rows[train_inds, :] = np.copy(y[train_inds, :])

    train_sim_mat = np.array(cosine_similarity(y_train_with_zero_rows) == 1, dtype=np.float32)

    print('zero out the test inds to use for training')
    train_zero_inds = np.where(y_train_with_zero_rows.sum(axis=1) == 0)[0] # includes both removed annotations and rows that were 0 to begin with
    train_sim_mat[train_zero_inds,:] = 0
    train_sim_mat[:, train_zero_inds] = 0 # zero out so that the unannotated proteins are not considered similar to each other

    train_labeled_inds = np.where(y_train_with_zero_rows.sum(axis=1) != 0)[0]

    return train_sim_mat, train_labeled_inds


def sim_reg_train_test(X, y, train_inds, test_inds, ker='rbf', X_pred=None):
    '''
    Builds and trains the autoencoder and scales the features.
    '''
    train_sim_mat, test_sim_mat = get_sim_mats(y, train_inds, test_inds) 

    input_dims = [X.shape[1]]
    encode_dims = [1000]
    
    model, history = build_sim_reg_model(X, input_dims, encode_dims, mtype='sim_reg_ae')
    export_history(history, model_name=model_name, kwrd='sim_reg_AE')

    mid_model = Model(inputs=model.input, outputs=model.get_layer('middle_layer').output)

    X = minmax_scale(mid_model.predict(X))

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

    y_score_trials = np.zeros((y.shape[1], n_trials), dtype=np.float)
    y_train = y[train_idx]
    y_test = y[test_idx]
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

    perf = dict()
    perf['pr_micro'] = pr_micro
    perf['pr_macro'] = pr_macro
    perf['fmax'] = fmax
    perf['acc'] = acc
    y_score_pred /= n_trials

    return perf, y_score_trials, y_score_pred


def create_sim_reg_batches_with_unlabeled(lap_mat_train, lap_mat_valid, X_train_noisy, X_train, X_valid_noisy, X_valid, labeled_vec_train, labeled_vec_valid, batch_size=32, num_epochs=5):
    # split X, Y and lap_mat into batches
    print('Creating sim reg batches with labeled vecs')
    inds_train = np.arange(X_train.shape[0])
    #labeled_vec_valid = np.ones((X_valid.shape[0], 1)) # assume all validation samples are labeled
    inds_test = np.arange(X_valid.shape[0])
    print(inds_train)
    train_epoch_in_list = [] # this will be a list of size num_epochs of lists of size number of batches for training
    train_epoch_out_list = []
    val_epoch_in_list = []
    val_epoch_out_list = []
    for epoch in range(0, num_epochs):
        np.random.shuffle(inds_train)
        batches = np.array_split(inds_train, (len(inds_train)/batch_size) + 1)
        print('Number of batches: ' + str(len(batches)))
        # generate batches before training
        print('Generating batches...')
        batch_time_start = time.time()
        train_batch_samples_in_list = []
        train_batch_samples_out_list = []
        for batch in batches:
            batch_labeled_vec = labeled_vec_train[batch, :]
            batch_lap_mat = lap_mat_train[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_lap_mat = np.concatenate([batch_labeled_vec, batch_lap_mat], axis=1)
            batch_samples_in = X_train_noisy[batch, :]
            batch_samples_out = [X_train[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            train_batch_samples_in_list.append(batch_samples_in)
            train_batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating batch for epoch ' + str(epoch) + '/' + str(num_epochs) + '. Time it took: ' + str(batch_time))
        val_batch_samples_in_list = []
        val_batch_samples_out_list = []
        print('Generating val batches...')
        batch_time_start = time.time()
        val_batches = np.array_split(inds_test, (len(inds_test)/batch_size) + 1)
        for batch in val_batches:
            batch_labeled_vec = labeled_vec_valid[batch, :]
            batch_lap_mat = lap_mat_valid[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_lap_mat = np.concatenate([batch_labeled_vec, batch_lap_mat], axis=1)
            batch_samples_in = X_valid_noisy[batch, :]
            batch_samples_out = [X_valid[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            val_batch_samples_in_list.append(batch_samples_in)
            val_batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating val batches for epoch ' + str(epoch) + '/' + str(num_epochs) + '. Time it took: ' + str(batch_time))
        train_epoch_in_list.append(train_batch_samples_in_list)
        train_epoch_out_list.append(train_batch_samples_out_list)
        val_epoch_in_list.append(val_batch_samples_in_list)
        val_epoch_out_list.append(val_batch_samples_out_list)

    return train_epoch_in_list, train_epoch_out_list, val_epoch_in_list, val_epoch_out_list
