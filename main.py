import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = pickle.load(open('X.pickle', 'rb'))
Y = pickle.load(open('Y.pickle', 'rb'))

# normalizing (RGB)
X = X / 255.0

# Input -> Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output

model = tf.keras.models.Sequential()

# conv layer gives each pixel a value
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))     # input_shape defines first layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))   # max pooling selects higest value processed by conv layer

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))   # RELU = max(0, input)
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=1, batch_size=32, validation_split=0.1)
# model.fit(X, Y, batch_size=32, epochs=3, validation_split=0.3)
