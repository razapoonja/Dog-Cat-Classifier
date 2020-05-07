import time
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

NAME = "Dogs-Cats-CNN-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

X = pickle.load(open('X.pickle', 'rb'))
Y = pickle.load(open('Y.pickle', 'rb'))

# normalizing (RGB)
X = X / 255.0

# Input -> Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output

model = Sequential([
        # conv layer gives each pixel a value
        # RELU = max(0, input)
        # input_shape defines first layer
        Conv2D(64, (3,3), activation='relu', input_shape= X.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),   # max pooling selects higest value processed by conv layer

        Conv2D(64, (3,3), activation='relu', input_shape= X.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3,3), activation='relu', input_shape= X.shape[1:]),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),    # converts 3d feature maps to 1d feature vectors

        Dense(64, activation='relu'),
        Dropout(0.2),    # prevents overfitting

        Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

model.fit(X, Y, 
        epochs=2, 
        batch_size=32, 
        validation_split=0.1,
        callbacks=[tensorboard])
