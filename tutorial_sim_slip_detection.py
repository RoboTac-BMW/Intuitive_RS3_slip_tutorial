import pickle
import numpy as np
import tensorflow as tf
from tf_utils import prepare_dataset
import time


def mySimpleRNN(rnn_units):
    model = [] # TODO
    return model

def mySimpleLSTM(cell_units):
    model = [] # TODO
    return model


def compute_loss(y, y_hat):
    # TODO
    pred = tf.reshape(y_hat, [y_hat.shape[0], -1])
    label = tf.reshape(y, [y.shape[0], -1])
    label = tf.cast(label, tf.float32)
    # Slightly downweight the loss for non-slip cases to reduce the false positives
    loss = []
    return tf.reduce_mean(loss)


# #######################################  II.1 Check the dataset ######################################################
data = pickle.load(open("slip_database.pickle", "rb"))

X = np.array(data['forces'])
y = np.array(data['slipping'])
cases = np.array(data['slip_label'])
objects = np.array(data['object_label'])

# Let's verify the dataset
unique_cond, count_cond = np.unique(cases, return_counts=True)
unique_objects, count_objects = np.unique(objects, return_counts=True)

# TODO: Check the shape of the dataset, count_cond, count_objects

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

########################################### Hyperparameters ####################################################
batch_size = 8
rnn_units = 64
learnig_rate = 0.5e-4

########################################II_2 Prepare the dataset #################################################
X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = prepare_dataset(X, y)
train_dataset = tf.data.Dataset.from_tensor_slices((X_trains, y_trains))
val_dataset = tf.data.Dataset.from_tensor_slices((X_vals, y_vals))
test_dataset = tf.data.Dataset.from_tensor_slices((X_tests, y_tests))

# Create batch
train_dataset = train_dataset.shuffle(100)
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.shuffle(50)
val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

dl_model = mySimpleRNN(rnn_units)
# dl_model = mySimpleLSTM(rnn_units)

optimizer = tf.keras.optimizers.Adam(learnig_rate)
epochs = 100
train_dir = 'results'
tf_summary_writer = tf.summary.create_file_writer(train_dir + '/' + str(time.time()))
train_step = 0
test_step = 0
epoch_loss_train = []
epoch_loss_val = []
with tf_summary_writer.as_default():
    for epoch in range(0, epochs):
        for (X_batch_train, y_batch_train) in train_dataset:
            # Use tf.GradientTape()
            with tf.GradientTape() as tape:
                y_batch_hat = dl_model()  # TODO
                loss = compute_loss()  # TODO
                epoch_loss_train.append(loss)
                tf.summary.experimental.set_step(train_step)
                tf.summary.scalar('train/ce_loss', loss)
                try:
                    grads = tape.gradient(loss, dl_model.trainable_weights)  # TODO
                except Exception as e:
                    print('Exception occured in computing gradients - ', e)
                    continue
                optimizer.apply_gradients(zip(grads, dl_model.trainable_variables))
            train_step += 1

        print("Epoch: ", epoch,  "Train Loss: ", np.mean(np.array(epoch_loss_train)))

        for (X_batch_val, y_batch_val) in val_dataset:
            y_batch_hat = dl_model(X_batch_val)
            loss = compute_loss(y_batch_val, y_batch_hat)  # TODO
            epoch_loss_val.append(loss)
            tf.summary.experimental.set_step(train_step)
            tf.summary.scalar('val/ce_loss', loss)

        print("Epoch: ", epoch,  "Val Loss: ", np.mean(np.array(epoch_loss_val)))

# TODO: add model saving options

test_dataset = tf.data.Dataset.from_tensor_slices((X_tests, y_tests))
test_dataset = test_dataset.batch(1, drop_remainder=True)
db_test_pred = []
db_test_label = []
db_test_ana = []
for (X_batch_test, y_batch_test) in test_dataset:
    y_batch_hat = dl_model(X_batch_test)
    # Add the tf condition for binary output
    y_batch_hat = tf.where(y_batch_hat > 0.5, tf.add(tf.multiply(y_batch_hat, 0), 1), tf.multiply(y_batch_hat, 0)) # Hmm, why is this added ?
    y_batch_ana = tf.where(X_batch_test[0, :, 1] == 0, tf.add(tf.multiply(X_batch_test[0, :, 1], 0), 1),
                           tf.multiply(X_batch_test[0, :, 1], 0))
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
print("Deep Learning approach - ")
print(confusion_matrix_dl.numpy())
print("Analytical approach - ")
print(confusion_matrix_ana.numpy())
