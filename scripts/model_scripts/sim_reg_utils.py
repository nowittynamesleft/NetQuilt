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


def build_sim_reg_AE(input_dim, encoding_dims, sim_reg_lamb, hidden_activation='tanh', lr=0.01):
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
    sgd = SGD(lr=lr, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=output_layers)

    losses = [sigmoid_sim_loss_label_mask, 'binary_crossentropy']
    loss_weights = [sim_reg_lamb, 1.]
    model.compile(optimizer=sgd, loss=losses, loss_weights=loss_weights)
    print (model.summary())

    return model


def train_sim_reg_model(model, lap_mat_train, lap_mat_valid, X_train_noisy, X_valid_noisy, X_train, X_valid, labeled_vec_train, labeled_vec_valid, sim_reg_lamb, batch_size=32, num_epochs=5):
    # split X, Y and lap_mat into batches
    inds_train = np.arange(X_train.shape[0])
    inds_test = np.arange(X_valid.shape[0])
    assert X_train_noisy.shape[0] == X_train.shape[0]
    assert X_valid_noisy.shape[0] == X_valid.shape[0]
    print(inds_train)
    print('training model')
    print('batch size:')
    print(batch_size)
    print('Sim reg lamb:')
    print(sim_reg_lamb)
    recon_loss_history = []
    recon_val_loss_history = []
    sim_loss_history = []
    sim_val_loss_history = []
    for epoch in range(0, num_epochs):
        time_start = time.time()
        np.random.shuffle(inds_train)
        batches = np.array_split(inds_train, (len(inds_train)/batch_size) + 1)
        num_train_batches = len(batches)
        print('Number of train batches: ' + str(num_train_batches))
        losses = np.array([0., 0.])
        # generate batches before training
        print('Generating batches...')
        batch_time_start = time.time()
        batch_samples_in_list = []
        batch_samples_out_list = []
        for batch in batches:
            batch_labeled_vec = labeled_vec_train[batch, :]
            batch_lap_mat = lap_mat_train[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_lap_mat = np.concatenate([batch_labeled_vec, batch_lap_mat], axis=1)
            batch_samples_in = X_train_noisy[batch, :]
            batch_samples_out = [X_train[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            batch_samples_in_list.append(batch_samples_in)
            batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating batches. Time it took: ' + str(batch_time))
        for i in range(0, len(batch_samples_in_list)):
            curr_losses = model.train_on_batch(batch_samples_in_list[i], batch_samples_out_list[i])
            losses[0] += curr_losses[1]
            losses[1] += curr_losses[2]
        val_batches = np.array_split(inds_test, (len(inds_test)/batch_size) + 1)
        num_val_batches = len(val_batches)
        print('Number of valid batches: ' + str(num_val_batches))
        val_losses = np.array([0., 0.])
        list_of_losses = []
        batch_samples_in_list = []
        batch_samples_out_list = []
        print('Generating val batches...')
        batch_time_start = time.time()
        for batch in val_batches:
            batch_labeled_vec = labeled_vec_valid[batch, :]
            batch_lap_mat = lap_mat_valid[batch, :]
            batch_lap_mat = batch_lap_mat[:, batch]
            batch_lap_mat = np.concatenate([batch_labeled_vec, batch_lap_mat], axis=1)
            batch_samples_in = X_valid_noisy[batch, :]
            batch_samples_out = [X_valid[batch, :]]
            batch_samples_out.insert(0, batch_lap_mat)
            batch_samples_in_list.append(batch_samples_in)
            batch_samples_out_list.append(batch_samples_out)
        batch_time_end = time.time()
        batch_time = batch_time_end - batch_time_start
        print('Done generating val batches. Time it took: ' + str(batch_time))
        for i in range(0, len(batch_samples_in_list)):
            curr_losses = model.test_on_batch(batch_samples_in_list[i], batch_samples_out_list[i])
            val_losses[0] += curr_losses[1] # curr_losses[1] is first individual loss, curr_losses[0] is total loss
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
