import numpy as np
import keras


class DataGenerator(Sequence):
    '''Generates data for neural network model'''
    def __init__(self, list_IDs, labels, 

