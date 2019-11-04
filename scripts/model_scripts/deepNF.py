from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Dense, GaussianNoise, Lambda
from keras.callbacks import EarlyStopping
import tensorflow as tf



def build_AE(input_dim, encoding_dims, hidden_activation='sigmoid'):
    """
    Function for building autoencoder.
    """
    # input layer
    print('Activation:')
    print(hidden_activation)
    input_layer = Input(shape=(input_dim, ))
    hidden_layer = input_layer
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='middle_layer')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,
                    activation='sigmoid')(hidden_layer)

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


def build_denoising_AE(input_dim, encoding_dims, hidden_activation='sigmoid'):
    """
    Function for building autoencoder.
    """
    # input layer
    print('Activation:')
    print(hidden_activation)
    input_layer = Input(shape=(input_dim, ))
    x = GaussianNoise(0.5)(input_layer)
    hidden_layer = Lambda(lambda a: tf.clip_by_value(a, 0, 1))(x)
    for i in range(0, len(encoding_dims)):
        # generate hidden layer
        if i == int(len(encoding_dims)/2):
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='middle_layer')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 activation=hidden_activation,
                                 name='layer_' + str(i+1))(hidden_layer)

    # reconstruction of the input
    decoded = Dense(input_dim,
                    activation='sigmoid')(hidden_layer)

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layer, outputs=decoded)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


def build_denoising_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        x = GaussianNoise(1.0)(input_layers[j])
        x = Lambda(lambda a: tf.clip_by_value(a, 0, 1))(x)
        hidden_layers.append(Dense(encoding_dims[0]/len(input_dims),
                                   # activity_regularizer=regularizers.l1(gamma[j]),
                                   activation='sigmoid'))(x)

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        hidden_layer = concatenate(hidden_layers)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == len(encoding_dims)/2:
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[-1]/len(input_dims),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model


def build_MDA(input_dims, encoding_dims):
    """
    Function for building multimodal autoencoder.
    """
    # input layers
    input_layers = []
    for dim in input_dims:
        input_layers.append(Input(shape=(dim, )))

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[0]/len(input_dims),
                                   # activity_regularizer=regularizers.l1(gamma[j]),
                                   activation='sigmoid')(input_layers[j]))

    # Concatenate layers
    if len(encoding_dims) == 1:
        hidden_layer = concatenate(hidden_layers, name='middle_layer')
    else:
        hidden_layer = concatenate(hidden_layers)

    # middle layers
    for i in range(1, len(encoding_dims)-1):
        if i == len(encoding_dims)/2:
            hidden_layer = Dense(encoding_dims[i],
                                 name='middle_layer',
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)
        else:
            hidden_layer = Dense(encoding_dims[i],
                                 # kernel_regularizer=regularizers.l1(1e-5),
                                 activation='sigmoid')(hidden_layer)

    if len(encoding_dims) != 1:
        # reconstruction of the concatenated layer
        hidden_layer = Dense(encoding_dims[0],
                             activation='sigmoid')(hidden_layer)

    # hidden layers
    hidden_layers = []
    for j in range(0, len(input_dims)):
        hidden_layers.append(Dense(encoding_dims[-1]/len(input_dims),
                                   activation='sigmoid')(hidden_layer))
    # output layers
    output_layers = []
    for j in range(0, len(input_dims)):
        output_layers.append(Dense(input_dims[j],
                                   activation='sigmoid')(hidden_layers[j]))

    # autoencoder model
    sgd = SGD(lr=0.01, momentum=0.9, decay=0.0, nesterov=False)
    model = Model(inputs=input_layers, outputs=output_layers)
    model.compile(optimizer=sgd, loss='binary_crossentropy')
    print (model.summary())

    return model
