import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd
import random

model = Sequential()
model.add(Dense(1, batch_input_shape=(10,1)))
model.summary()

opt = sgd(lr=1e-3)
model.compile(optimizer=opt, loss='mse')

X = np.array([[i,j] for i in range(20) for j in range(20)])
Y = np.array([2*i for i in range(20)])
print (X)
print (Y)

model.fit(X, Y, batch_size=10, nb_epoch=20,verbose=1)

X_test = np.array([random.randint(0,40) for i in np.arange(10)])
print (X_test)

pred = model.predict(X_test, batch_size=10)
for idx in np.arange(X_test.shape[0]):
    print ('Model predicts {} for input {}'.format(int(pred[idx]+.5), X_test[idx]))
