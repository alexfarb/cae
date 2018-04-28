from keras.models import Model
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input
import numpy as np

inp =  Input(shape=(5, 1))
conv = Conv1D(filters=2, kernel_size=2)(inp)
pool = MaxPool1D(pool_size=2)(conv)
flat = Flatten()(pool)
dense = Dense(1)(flat)
model = Model(inp, dense)
model.compile(loss='mse', optimizer='adam')

# print(model.summary())

# get some data
X = np.expand_dims(np.random.randn(10, 5), axis=2)
print(X)
y = np.random.randn(10, 1)

# fit model
model.fit(X, y)