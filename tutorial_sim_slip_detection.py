import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import time

def mySimpleMLP(units, horizon, window):
    # TODO: Implement a simple MLP model, with windowing approach
    model = []
    return model

def mySimpleRNN(rnn_units):
    # TODO: Implement a RNN model with LSTM
    model = []
    return model


def compute_loss(y, y_hat):
    # TODO
    scale = 100
    pred = tf.reshape(y_hat, [y_hat.shape[0], -1])
    label = tf.reshape(y, [y.shape[0], -1])
    label = tf.cast(label, tf.float32)
    # Slightly downweight the loss for non-slip cases to reduce the false positives
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


data = pickle.load(open("slip_database.pickle", "rb"))

X = np.array(data['forces'])
y = np.array(data['slipping'])
cases = np.array(data['slip_label'])
objects = np.array(data['object_label'])

# Let's verify the dataset
unique_cond, count_cound = np.unique(cases, return_counts=True)
unique_objects, count_objects = np.unique(objects, return_counts=True)

'''
# Inspect the data - slip and no slip
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
'''

X_trains, X_tests, y_trains, y_tests = prepare_dataset(X, y)
train_dataset = tf.data.Dataset.from_tensor_slices((X_trains, y_trains))
test_dataset = tf.data.Dataset.from_tensor_slices((X_tests, y_tests))

# Create batch
train_dataset = train_dataset.shuffle(100)
train_dataset = train_dataset.batch(8, drop_remainder=True)
test_dataset = test_dataset.batch(8, drop_remainder=True)

# TODO: try different unit size of the RNN
dl_model = mySimpleRNN(1024)
# dl_model = mySimpleMLP(1024)
# TODO: try different learning rate
optimizer = tf.keras.optimizers.Adam(0.5e-4)
# TODO: when should we stop training ?
epochs = 50
train_dir = 'results'
tf_summary_writer = tf.summary.create_file_writer(train_dir + '/' + str(time.time()))
train_step = 0
test_step = 0

with tf_summary_writer.as_default():
    for epoch in range(0, epochs):
        for (X_batch_train, y_batch_train) in train_dataset:
            # Use tf.GradientTape()
            with tf.GradientTape() as tape:
                y_batch_hat = []  # TODO
                # y_batch_hat = tf.cond(y_batch_hat > 0.5, lambda: tf.add(tf.multiply(y_batch_hat, 0), 1), lambda: tf.multiply(y_batch_hat, 0))
                loss = []  # TODO
                tf.summary.experimental.set_step(train_step)
                tf.summary.scalar('train/ce_loss', loss)
                if train_step % 100 == 0:
                    print("Epoch: ", epoch, " Step: ", train_step, "Train Loss: ", loss.numpy())
                try:
                    grads = tape.gradient(loss, dl_model.trainable_weights) # TODO
                except Exception as e:
                    print('Exception occured in computing gradients - ', e)
                    continue
                optimizer.apply_gradients(zip(grads, dl_model.trainable_variables))
            train_step += 1

        for (X_batch_test, y_batch_test) in test_dataset:
            y_batch_hat = dl_model(X_batch_test)
            loss = compute_loss(y_batch_test, y_batch_hat)  # TODO
            if test_step % 100 == 0:
                print("Epoch: ", epoch, " Step: ", test_step, "Val Loss: ", loss.numpy())
            tf.summary.experimental.set_step(train_step)
            tf.summary.scalar('val/ce_loss', loss)
            test_step += 1

# TODO: add model saving options
# Run the inference on the trainined model

test_dataset = tf.data.Dataset.from_tensor_slices((X_tests, y_tests))
test_dataset = test_dataset.batch(1, drop_remainder=True)
db_test_pred = []
db_test_label = []
db_test_ana = []
for (X_batch_test, y_batch_test) in test_dataset:
    y_batch_hat = dl_model(X_batch_test)
    # Add the tf condition for binary output
    y_batch_hat = tf.where(y_batch_hat > 0.5, tf.add(tf.multiply(y_batch_hat, 0), 1),tf.multiply(y_batch_hat, 0))
    # TODO: add your own analytical slip detection step implemented here !!
    y_batch_ana = tf.where(X_batch_test[0, :, 1] == 0, tf.add(tf.multiply(X_batch_test[0, :, 1], 0), 1), tf.multiply(X_batch_test[0, :, 1], 0))
    # y_batch_ana = []

    pred = tf.reshape(y_batch_hat, [y_batch_hat.shape[0], -1])
    label = tf.reshape(y_batch_test, [y_batch_test.shape[0], -1])
    db_test_pred.append(pred.numpy())
    db_test_label.append(label.numpy())
    db_test_ana.append(y_batch_ana)


# Generate the confusion matrix and some metrics post training
db_test_pred = np.array(db_test_pred)
db_test_label = np.array(db_test_label)
db_test_ana = np.array(db_test_ana)
db_test_ana = np.reshape(db_test_ana, -1)
db_test_pred = np.reshape(db_test_pred, -1)
db_test_label = np.reshape(db_test_label, -1)
confusion_matrix_dl = tf.math.confusion_matrix(db_test_label, db_test_pred)
confusion_matrix_ana = tf.math.confusion_matrix(db_test_label, db_test_ana)

# TODO: Better formating and nice plots

print(confusion_matrix_dl.numpy())
print(confusion_matrix_ana.numpy())
