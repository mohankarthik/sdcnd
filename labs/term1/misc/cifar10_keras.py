import numpy as np

import keras
from keras.datasets import cifar10
import tensorflow

# Import the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()