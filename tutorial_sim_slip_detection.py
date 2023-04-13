import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time


def mySimpleRNN(rnn_units):
    # TODO: Fill in the model
    model = []
    return model


def compute_loss(y, y_hat):
    scale = 100
    # TODO: compute cross-entropy loss
    pred = tf.reshape(y_hat, [y_hat.shape[0], -1])
    label = tf.reshape(y, [y.shape[0], -1])
    label = tf.cast(label, tf.float32)
    # pred = tf.clip_by_value(pred, 0, 1.)
    loss = []
    return tf.reduce_mean(loss)*scale


def train_test_split(X, y, ratio=0.8):
   split=int(len(X)*ratio)
   X_train = X[:split]
   y_train = y[:split]

   X_test = X[split:]
   y_test = y[split:]

   return X_train, X_test, y_train, y_test


def prepare_dataset(X, y):
    # Distribute slip and no slip samples
    X_trains = []
    X_tests = []
    y_trains = []
    y_tests = []
    sample_indices = np.arange(0, len(X), len(X)/6)
    sample_indices = np.append(sample_indices, len(X))
    for i in range(0, len(sample_indices)-1):
        X_train, X_test, y_train, y_test = train_test_split(X[int(sample_indices[i]):int(sample_indices[i+1])], y[int(sample_indices[i]):int(sample_indices[i+1])])
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

    X_trains = np.array(X_trains)
    X_trains = X_trains.reshape(X_trains.shape[0]*X_trains.shape[1], X_trains.shape[2], X_trains.shape[3])
    X_tests = np.array(X_tests)
    X_tests = X_tests.reshape(X_tests.shape[0]*X_tests.shape[1], X_tests.shape[2], X_tests.shape[3])
    y_trains = np.array(y_trains)
    y_trains = y_trains.reshape(y_trains.shape[0]*y_trains.shape[1], y_trains.shape[2])
    y_tests = np.array(y_tests)
    y_tests = y_tests.reshape(y_tests.shape[0]*y_tests.shape[1], y_tests.shape[2])

    return X_trains, X_tests, y_trains, y_tests

# ..............########################  Main Code ######################.....................


data = pickle.load(open("slip_database_ChipsCan.pickle", "rb"))

X = np.array(data['forces'])
y = np.array(data['slipping'])


# TODO: Inspect the data - slip and no slip
fig, ax = plt.subplots(2, 4)
ax[0, 0].plot(X[0, :, 0], '.r')
ax[1, 0].plot(X[-1, :, 0], '*r')
ax[0, 1].plot(X[0, :, 1], '.r')
ax[1, 1].plot(X[-1, :, 1], '*r')
ax[0, 2].plot(X[0, :, 2], '.r')
ax[1, 2].plot(X[-1, :, 2], '*r')
ax[0, 3].plot(y[0, :], '.b')
ax[1, 3].plot(y[-1, :], '*b')
plt.show()


X_trains, X_tests, y_trains, y_tests = prepare_dataset(X, y)
train_dataset = tf.data.Dataset.from_tensor_slices((X_trains, y_trains))
test_dataset = tf.data.Dataset.from_tensor_slices((X_tests, y_tests))

# Create batch
train_dataset = train_dataset.shuffle(50)
train_dataset = train_dataset.batch(8, drop_remainder=True)
test_dataset = test_dataset.batch(1, drop_remainder=True)

# TODO: Try different rnn units
rnn_model = mySimpleRNN(rnn_units=128)
optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 10
train_dir = 'results'
tf_summary_writer = tf.summary.create_file_writer(train_dir + '/' + str(time.time()))
train_step = 0
test_step = 0

with tf_summary_writer.as_default():
    for epoch in range(0, epochs):
        for (X_batch_train, y_batch_train) in train_dataset:
            # Use tf.GradientTape()
            with tf.GradientTape() as tape:
                y_batch_hat = []  # TODO: pass the input through the model
                loss = []  # TODO
                tf.summary.experimental.set_step(train_step)
                tf.summary.scalar('train/ce_loss', loss)
                if train_step % 25:
                    print("Epoch: ", epoch, " Step: ", train_step, "Train Loss: ", loss)
                try:
                    grads = tape.gradient(loss, rnn_model.trainable_weights) # TODO
                except Exception as e:
                    print('Exception occured in computing gradients - ', e)
                    continue
                optimizer.apply_gradients(zip(grads, rnn_model.trainable_variables))
            train_step += 1

        for (X_batch_test, y_batch_test) in test_dataset:
            y_batch_hat = rnn_model(X_batch_test)
            loss = compute_loss(y_batch_test, y_batch_hat)  # TODO
            if test_step % 25:
                print("Epoch: ", epoch, " Step: ", train_step, "Test Loss: ", loss)
            tf.summary.experimental.set_step(train_step)
            tf.summary.scalar('test/ce_loss', loss)
            test_step += 1

