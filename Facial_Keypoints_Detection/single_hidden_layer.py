import numpy as np
import matplotlib.pyplot as plt
import os
import time
import logging

from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nolearn.lasagne import BatchIterator
import tensorflow as tf

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'
FLOOKUP = 'data/IdLookupTable.csv'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] % (levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H: %M:%S',
                    filename='myapp.log',
                    filemode='w')


def load(test=False, cols=None):
    """
    Loads the dataset.
    Parameters
    :param test: optional, defaults to 'False'
                 Flag indicating if we need to load from 'FTEST'('True') or 'FTRAIN' ('False')
    :param cols: optional, defaults to 'None'
                 A list of columns you're interested in. If specified only reterns these columns.
    :return: A tuple of X and y, if 'test' was set to 'True' y contains 'None'. 
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # load pandas dataframe

    # The Image column has pixel values separated by space;
    # convert the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # prints the number of values for each column
    print(df.count())
    # logging.info('The number of values for each column ', df.count())
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[: -1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None
    return X, y


def plot_sample(x, y, axis):
    """
    Plots a single sample image with keypoints on top.
    :param x: Image data.
    :param y: Keypoints to plot.
    :param axis: Plot over which to draw the sample.
    :return: None
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2]*48 + 48, y[1::2]*48 + 48, marker='x', s=10)


def fully_connected(input, size):
    """
    Create a fully connected TensorFlow layer.
    :param input: Input tensor for calculating layer shape.
    :param size: Layer size, e.g. number of units
    :return: A graph variable calculating single fully connected layer.
    """
    weights = tf.get_variable('weights',
                              shape=[input.get_shape()[1], size],
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('bias',
                           shape=[size],
                           initializer=tf.constant_initializer(0.0)
                           )
    return tf.matmul(input, weights) + bias


def model_pass(input, output_size=30):
    """
    Performs a whole model pass.
    :param input: Input tensor to be passed through the model.
    :return: Model prediction.  
    """
    with tf.variable_scope('hidden'):
        hidden = fully_connected(input, 100)
    relu_hidden = tf.nn.relu(hidden)
    with tf.variable_scope('out'):
        prediction = fully_connected(relu_hidden, size=output_size)
    return prediction


def calc_loss(predictions, labels):
    """
    Calculates loss with Numpy.
    :param predictions: ndarray. Predictions.
    :param labels: ndarrya. Actual values.
    :return: Squared mean error for given predictions.
    """
    return np.mean(np.square(predictions-labels))


def get_time_hhmmss(start):
    """
    Calculates time since 'start' and formats as a string.
    :param start: Time starting point.
    :return: Nicely formatted time difference between now and 'start'.
    """
    end = time.time()
    m, s = divmod(end-start, 60)
    h, m = divmod(m, 60)
    time_str = '%02d:%02d:%02d' % (h, m, s)
    return time_str


# Plots history of learning curves for a specific model. You may want to call 'pyplot.show()' afterwords.
def plot_learning_curves(root_location, model, linewidth = 2, train_linestyle='b-', valid_linestyle='g-'):
    """
    Plots history of learning curves for a specific model based on the saved training history.
    You may want to call 'pyplot.show()' aferwards.
    :param model: optional, defaults to current model name, Model name.
    :param linewidth: optional, defaults to 2, Line thickness.
    :param train_linestyle: optional, defaults to 'b-', Matplotlib line style for the training curve.
    :param valid_linestyle: optional, defaults to 'g-', Matplotlib line style for the validation curve.
    :return: Number of epochs plotted.
    """
    model_history = np.load(root_location + model + '/train_history.npz')
    train_loss = model_history['train_loss_history']
    valid_loss = model_history['valid_loss_history']
    epochs = train_loss.shape[0]
    x_axis = np.arange(epochs)
    plt.plot(x_axis[train_loss > 0], train_loss[train_loss > 0], train_linestyle,
             linewidth=linewidth, label=model+'train')
    plt.plot(x_axis[valid_loss > 0], valid_loss[valid_loss > 0], valid_linestyle,
             linewidth=linewidth, label=model+'valid')
    return epochs


def single_hidden_layer():
    """
    Run the graph to train the model and predict the test data.
    :return: 
    """
    # Load dataset
    X, y = load()
    img = X[11].reshape(96, 96)
    plt.imshow(img, cmap='gray')
    # plt.show()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5)

    # Predefined parameters
    image_size = 96
    num_keypoints = 30
    batch_size = 36
    num_epochs = 1001
    learning_rate = 0.01
    momentum = 0.9

    model_name = "1fc_b" + str(batch_size) + "_e" + str(num_epochs - 1)
    model_variable_scope = model_name
    root_location = 'models/'
    model_path = root_location + model_name + '/model.ckpt'
    train_history_path = root_location + model_name + '/train_history'

    os.makedirs(root_location + model_name + '/', exist_ok=True)

    # Training
    graph = tf.Graph()

    with graph.as_default():
        # Input data. For the training data, we use a placeholder that will be fed at run time with a training minibatch
        tf_x_batch = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
        tf_y_batch = tf.placeholder(tf.float32, shape=(None, num_keypoints))

        # Training computation.
        with tf.variable_scope(model_variable_scope):
            predictions = model_pass(tf_x_batch, num_keypoints)

        def get_predictions_in_batches(X, session):
            """
            Calculates predictions in batches of 128 examples at a time, using `session`'s calculation graph.

            Parameters
            ----------
            X       : ndarray
                      Dataset to get predictions for.
            session :
                      TensorFlow session to be used for predicting. Is expected to have a `predictions` var 
                      in the graph along with a `tf_x_batch` placeholder for incoming data.

            Returns
            -------
            N-dimensional array of predictions.
            """
            p = []
            batch_iterator = BatchIterator(batch_size=128)
            for x_batch, _ in batch_iterator(X):
                [p_batch] = session.run([predictions], feed_dict={
                    tf_x_batch: x_batch
                }
                                        )
                p.extend(p_batch)
            return p

        loss = tf.reduce_mean(tf.square(predictions - tf_y_batch))

        # Optimizer.
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=True
        ).minimize(loss)

        start = time.time()
        every_epoch_to_log = 5

        with tf.Session(graph=graph) as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_loss_history = np.zeros(num_epochs)
            valid_loss_history = np.zeros(num_epochs)
            print("============TRAINING============")
            # logging.info('============TRAINING============')
            for epoch in range(num_epochs):
                # Train on whole randomised dataset in batches
                batch_iterator = BatchIterator(batch_size=batch_size, shuffle=True)
                for x_batch, y_batch in batch_iterator(x_train, y_train):
                    session.run([optimizer], feed_dict={
                        tf_x_batch: x_batch,
                        tf_y_batch: y_batch
                    }
                                )

                # If another significant epoch ended, we log our losses.
                if (epoch % every_epoch_to_log) == 0:
                    # Get training data predictions and log training loss:
                    train_loss = calc_loss(
                        get_predictions_in_batches(x_train, session),
                        y_train
                    )
                    train_loss_history[epoch] = train_loss

                    # Get validation data predictions and log validation loss:
                    valid_loss = calc_loss(
                        get_predictions_in_batches(x_valid, session),
                        y_valid
                    )
                    valid_loss_history[epoch] = valid_loss

                    if (epoch % 100) == 0:
                        print('-----EPOCH %4d/%d' % (epoch, num_epochs))
                        print('   Train loss: %.8f' % train_loss)
                        print('Validation loss: %.8f' % valid_loss)
                        print('       Time:' + get_time_hhmmss(start))
                        # logging.info('-----EPOCH %4d/%d' % (epoch, num_epochs))
                        # logging.info('   Train loss: %.8f' % train_loss)
                        # logging.info('Validation loss: %.8f' % valid_loss)
                        # logging.info('       Time:' + get_time_hhmmss(start))

            # Evaluate on test dataset.
            test_loss = calc_loss(
                get_predictions_in_batches(x_test, session),
                y_test
            )
            print('==========================================')
            print('Test score: %.3f(loss = %.8f)' % (np.sqrt(test_loss) * 48.0, test_loss))
            print('Total time: ' + get_time_hhmmss(start))
            # logging.info('==========================================')
            # logging.info('Test score: %.3f(loss = %.8f' % (np.sqrt(test_loss)*48.0, test_loss))
            # logging.info('Total time: ' + get_time_hhmmss(start))
            save_path = saver.save(session, model_path)
            print('Model file:' + save_path)
            # logging.info('Model file:' + save_path)
            np.savez(train_history_path, train_loss_history=train_loss_history, valid_loss_history=valid_loss_history)
            print('Train history file: ' + train_history_path)
            # logging.info('Train history file: ' + train_history_path)

    new_model_epochs = plot_learning_curves(root_location, model_name)
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0.0005, 0.01)
    plt.xlim(0, new_model_epochs)
    plt.yscale('log')
    # plt.show()

    X, _ = load(test=True)

    with graph.as_default():
        tf_x = tf.constant(X)
        with tf.variable_scope(model_variable_scope, reuse=True):
            tf_p = model_pass(tf_x)

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        load_path = saver.restore(session, model_path)
        p = tf_p.eval()

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], p[i], ax)

    plt.show()


single_hidden_layer()

