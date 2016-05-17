import tensorflow as tf
import numpy as np
import imageRead
import sys

from skimage import exposure, filters
from freezeGraph import *

# Globals
LOAD_MODEL = False
TRAIN_MODEL = True
FREEZE_GRAPH = True


BATCH_SIZE = 100
STEP_SIZE_MAX = 10000
STEP_SIZE_PRINT = 100
STEP_SIZE_SAVE = 1000

TRAINING_RATIO = 0.9
DISTORTION_RATE = 1.0
ADD_FLIPPED = True

CAR_ORIGIN_POS = [376.0, 480.0]

# Set parameters
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 0.1

# load the input data
newHeight = 120
newWidth = 180
resize = True
data_input = imageRead.combine_data(
    resize,
    newHeight,
    newWidth,
    training_ratio=TRAINING_RATIO,
    distortionRate=DISTORTION_RATE,
    carOriginPos=CAR_ORIGIN_POS,
    addFlipped=ADD_FLIPPED
)

# Functions for weight and bias initialisation
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Convolution and pooling functions
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Loss functions
def angle_loss(v1, v2):
    v1_magnitude = tf.sqrt(tf.reduce_sum(tf.square(v1), 1))
    v2_magnitude = tf.sqrt(tf.reduce_sum(tf.square(v2), 1))
    dot_product = tf.reduce_sum(tf.mul(v1, v2), 1)
    return dot_product / (v1_magnitude * v2_magnitude)

def euclidian_loss(p1, p2):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(p1, p2)), 1))

# Evaluation function
def eval_accuracy(loss_func, dataset):
    return loss_func.eval(
        feed_dict = {
            x: dataset.images,
            y: dataset.labels,
            keep_prob: 1.0
        }
    )

# Building graph and training model.    
if TRAIN_MODEL:
    print("\nBuilding Graph..")

    with tf.Session() as sess:
        
        # Initialize constants.
        img_center_pos = tf.constant(CAR_ORIGIN_POS)

        # Create placeholders
        x = tf.placeholder(tf.float32, shape=[None, newHeight*newWidth], name='input_images')
        y = tf.placeholder(tf.float32, shape=[None, 2], name='input_labels')
        keep_prob = tf.placeholder('float', name='keep_prob')

        # Build the first convolution layer
        x_image = tf.reshape(x, [-1, newHeight, newWidth, 1])
        W_conv1 = weight_variable([11, 11, 1, 64])
        b_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 3, 3, 1], padding='SAME') + b_conv1)
        h_pool1 = max_pool_3x3(h_conv1)

        # Build the second convolution layer
        W_conv2 = weight_variable([5, 5, 64, 128])
        b_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        # Build the third convolution layer
        W_conv3 = weight_variable([3, 3, 128, 256])
        b_conv3 = bias_variable([256])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        # Build the fourth convolution layer
        W_conv4 = weight_variable([3, 3, 256, 256])
        b_conv4 = bias_variable([256])
        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)

        # Build the fifth convolution layer
        W_conv5 = weight_variable([3, 3, 256, 512])
        b_conv5 = bias_variable([512])
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

        # Add a fully connected layer
        W_fc1 = weight_variable([5*4*512, 1024])
        b_fc1 = bias_variable([1024])
        h_conv5_flat = tf.reshape(h_conv5, [-1, 5*4*512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

        # Add dropout to reduce overfitting
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Add a second fully connected layer
        W_fc2 = weight_variable([1024, 1024])
        b_fc2 = bias_variable([1024])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

        # Add a readout layer
        W_ro = weight_variable([1024, 2])
        b_ro = bias_variable([2])
        y_conv = tf.add(tf.matmul(h_fc2_drop, W_ro), b_ro, name='output')

        # Add post processing layer
        alphas = tf.reshape(tf.abs(tf.div(CAR_ORIGIN_POS[1], tf.sub(y_conv[:, 1], CAR_ORIGIN_POS[1]))), [-1, 1])
        y_conv_processed = tf.add(tf.mul(alphas, tf.sub(y_conv, CAR_ORIGIN_POS)), CAR_ORIGIN_POS, name='output_processed')

        # Loss functions. Euclidian distance select as default.
        loss_euc = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y - y_conv), 1)), name='loss_euclidian')
        loss_ang = tf.sub(1.0, tf.reduce_mean(angle_loss(tf.sub(y, img_center_pos), tf.sub(y_conv, img_center_pos))), name='loss_angular')

        # Train and test the network.
        global_step = tf.Variable(0, trainable=False)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss_euc, global_step=global_step)

        # Initialize variables and save graph.
        tf.initialize_all_variables().run()
        tf.train.write_graph(sess.graph.as_graph_def(), './models', 'graph.pb', as_text=True)

        # Initialize saver object.
        saver = tf.train.Saver()

        # Loads previously trained model from checkpoint file.
        if LOAD_MODEL:
            print('\nLoading model..')

            try:
                saver.restore(sess, "./models/checkpoint-" + str(STEP_SIZE_MAX))
            except ValueError:
                sys.exit("\nError! Could not load model!\n")


        # Test Accuracy
        print('')
        print('Mean Test Accuracy (euclidian distance): ' + str(eval_accuracy(loss_euc, data_input.test)))
        print('Mean Test Accuracy (angle degrees): ' + str(np.arccos(1.0 - eval_accuracy(loss_ang, data_input.test)) * (180.0 / np.pi)))
        print('')
        

        # Training Step
        print('Training...')
        for step in range(STEP_SIZE_MAX + 1):

            batch = data_input.train.next_batch(BATCH_SIZE)
            sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

            if step % STEP_SIZE_PRINT == 0:
                batch = data_input.test.next_batch(BATCH_SIZE)
                acc = loss_euc.eval(feed_dict={x:batch[0], y: batch[1], keep_prob: 1.0})
                print("Test accuracy after step %s: %s" % (step, acc))
            if step % STEP_SIZE_SAVE == 0:
                save_path = saver.save(sess, "./models/checkpoint", global_step=step, latest_filename='checkpoint_state')
                print("Model saved in file: %s" % save_path)



        # Test Accuray
        print('')
        print('Mean Test Accuracy (euclidian distance): ' + str(eval_accuracy(loss_euc, data_input.test)))
        print('Mean Test Accuracy (angle degrees): ' + str(np.arccos(1.0 - eval_accuracy(loss_ang, data_input.test)) * (180.0 / np.pi)))
        print('')
        




# Freeze graph by combining graph structure file with checkpoint file.
if FREEZE_GRAPH:
    
    input_graph = './models/graph.pb'
    input_saver = ""
    input_binary = False
    input_checkpoint = './models/checkpoint' + "-" + str(STEP_SIZE_MAX)
    #input_checkpoint = './models/model.ckpt' + "-11000"

    output_node_names = "output,output_processed,loss_euclidian,loss_angular"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph = './models/graph_freezed.pb'
    clear_devices = False
    initializer_nodes = ''

    print("Freezing graph: " + input_graph + ' with checkpoint: ' + input_checkpoint)
    print("")

    freeze_graph(
        input_graph,
        input_saver,
        input_binary,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes
    )
        

'''

    EXAMPLE LOADING FREEZED GRAPH

    with tf.Graph().as_default():
        graph_def = tf.GraphDef()
        with open('./models/graph_freezed.pb', 'rb') as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        with tf.Session() as sess:

            # Original Predictions
            output_node = sess.graph.get_tensor_by_name('import/output:0')
            predictions = output_node.eval(
                feed_dict = {
                    'import/input_images:0': data_input.test.images,
                    'import/keep_prob:0': 1.0
                }
            )
            print("Test Set Example Predictions (0-10):")
            print(predictions[0:10])
            print("")

            # Processed Predictions
            output_node_processed = sess.graph.get_tensor_by_name('import/output_processed:0')
            predictions = output_node_processed.eval(
                feed_dict = {
                    'import/input_images:0': data_input.test.images,
                    'import/keep_prob:0': 1.0
                }
            )
            print("Test Set Example Predictions Processed (0-10):")
            print(predictions[0:10])
            print(" ")

            # Mean loss euclidian
            loss_euc = sess.graph.get_tensor_by_name('import/loss_euclidian:0')
            mean_loss = loss_euc.eval(
                feed_dict = {
                    'import/input_images:0': data_input.test.images,
                    'import/input_labels:0': data_input.test.labels,
                    'import/keep_prob:0': 1.0
                }
            )
            print('Test Set Accuracy (euclidian): ' + str(mean_loss))
            print("")

            # Mean loss angle
            loss_ang = sess.graph.get_tensor_by_name('import/loss_angular:0')
            mean_loss = loss_ang.eval(
                feed_dict = {
                    'import/input_images:0': data_input.test.images,
                    'import/input_labels:0': data_input.test.labels,
                    'import/keep_prob:0': 1.0
                }
            )
            print('Test Set Accuracy (angle degrees): ' + str(np.arccos(1.0 - mean_loss) * (180.0 / np.pi)))
            print("")

'''
