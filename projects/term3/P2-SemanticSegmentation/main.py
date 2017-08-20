#!/usr/bin/python3
"""
@file: main.py
@History:
  # 2017/08/17: Initial version - nemesis.nitt@gmail.com
"""
import os
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import csv
import datetime
import scipy.misc


def load_vgg(sess, vgg_path):
    """
    Load Pre-trained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # load the model from the given vgg_path
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

    # extract the layers of the vgg to modify into a FCN
    graph = tf.get_default_graph()
    img_imp = graph.get_tensor_by_name('image_input:0')
    keep = graph.get_tensor_by_name('keep_prob:0')
    layer3 = graph.get_tensor_by_name('layer3_out:0')
    layer4 = graph.get_tensor_by_name('layer4_out:0')
    layer7 = graph.get_tensor_by_name('layer7_out:0')
    return img_imp, keep, layer3, layer4, layer7


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.
    Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # FCN-8 - Decoder
    # To build the decoder portion of FCN-8, we’ll upsample the input to the
    # original image size.  The shape of the tensor after the final
    # convolutional transpose layer will be 4-dimensional:
    #    (batch_size, original_height, original_width, num_classes).

    # Making sure the resulting shape are the same
    vgg_layer7_logits = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=1, name='vgg_layer7_logits')
    vgg_layer4_logits = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=1, name='vgg_layer4_logits')
    vgg_layer3_logits = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=1, name='vgg_layer3_logits')

    # Let’s implement those transposed convolutions we discussed earlier as follows:
    fcn_decoder_layer1 = tf.layers.conv2d_transpose(
        vgg_layer7_logits, num_classes, kernel_size=4, strides=(2, 2),
        padding='same', name='fcn_decoder_layer1')

    # Then we add the first skip connection from the vgg_layer4_out
    fcn_decoder_layer2 = tf.add(fcn_decoder_layer1, vgg_layer4_logits, name='fcn_decoder_layer2')

    # We can then follow this with another transposed convolution layer making sure the resulting shape are the same
    # as layer3
    fcn_decoder_layer3 = tf.layers.conv2d_transpose(
        fcn_decoder_layer2, num_classes, kernel_size=4, strides=(2, 2),
        padding='same', name='fcn_decoder_layer3')

    # We’ll repeat this once more with the third pooling layer output.
    fcn_decoder_layer4 = tf.add(fcn_decoder_layer3, vgg_layer3_logits, name='fcn_decoder_layer4')

    # The final layer
    fcn_decoder_layer5 = tf.layers.conv2d_transpose(
        fcn_decoder_layer4, num_classes, kernel_size=16, strides=(8, 8),
        padding='same', name='fcn_decoder_layer5')

    # Return the final fcn output
    return fcn_decoder_layer5


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # Reshape the 4D output and label tensors to 2D, so each row represent a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    # Now define a loss function and a trainer/optimizer
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss


def train_nn(
        sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss,
        input_image, correct_label, keep_prob, learning_rate,
        runs_dir=None, data_dir=None, image_shape=None, logits=None, save_epoch=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data. Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: The actual learning rate in float
    :param runs_dir: directory where model weights and samples will be saved
    :param data_dir: directory where the Kitty dataset is stored
    :param image_shape: shape of the input image for prediction
    :param logits: TF Placeholder for the FCN prediction
    :param save_epoch: Number of epochs after which to save
    """
    # Logger
    log_writer = None
    log_file = None
    if runs_dir is not None:
        # Make sure that the root folder exists
        assert os.path.exists(runs_dir)

        # Create a sub folder for this run after ensuring that it does not exist
        runs_dir = os.path.join(runs_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        assert not os.path.exists(runs_dir)
        os.makedirs(runs_dir)

        # Start a log file
        log_filename = os.path.join(runs_dir, "fcn_training_progress.csv")
        log_fields = ['epoch', 'exec_time', 'training_loss']
        log_file = open(log_filename, 'w')
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        log_writer.writeheader()
        log_file.flush()

    # Start the global time
    total_start_time = time.clock()

    # Initialize the globals
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        training_loss = 0
        training_samples = 0

        # periodically save every 25 epoch runs
        if data_dir is not None and save_epoch is not None and runs_dir is not None and (i % save_epoch) == 0:
            # Save inference data using save_inference_samples
            save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, i)

        # start epoch training timer
        print("Running epochs:", i+1)
        start_time = time.clock()

        # train on batches
        for X, y in get_batches_fn(batch_size):
            training_samples += len(X)
            loss, _ = sess.run(
                [cross_entropy_loss, train_op],
                feed_dict={input_image: X, correct_label: y, keep_prob: 0.8})
            training_loss += loss

        # Calculate training loss
        training_loss /= training_samples
        end_time = time.clock()
        training_time = end_time - start_time
        print("Epoch {} execution took {} seconds, with training loss: {}".format(i, training_time, training_loss))

        # log if doing real training
        if runs_dir is not None:
            log_writer.writerow(
                {
                    'epoch': i+1,
                    'exec_time': training_time,
                    'training_loss': training_loss
                })
            log_file.flush()

    # Compute the total time taken
    total_end_time = time.clock()
    total_time = total_end_time - total_start_time
    print("total execution took {} seconds".format(total_time))


def save_inference_samples(
        runs_dir, data_dir, sess, image_shape, logits,
        keep_prob, input_image, epoch):
    """
    save model weights and generate samples.
    :param runs_dir: directory where model weights and samples will be saved
    :param data_dir: directory where the Kitty dataset is stored
    :param sess: TF Session
    :param image_shape: shape of the input image for prediction
    :param logits: TF Placeholder for the FCN prediction
    :param keep_prob: TF Placeholder for dropout keep probability
    :param input_image: TF Placeholder for input images
    :param epoch: Number of epochs or Final label
    """
    # Make folder for current epoch
    output_dir = os.path.join(runs_dir, str(epoch))
    assert not os.path.exists(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Epoch {} finished. Saving test images to: {}'.format(epoch, output_dir))
    image_outputs = helper.gen_test_output(sess, logits, keep_prob, input_image,
                                           os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

    # Save the model
    saver = tf.train.Saver()
    filefcn_path = os.path.join(output_dir, 'model.ckpt'.format(epoch))
    saver.save(sess, filefcn_path)
    print('Model saved to: {}'.format(filefcn_path))


def run():
    """
    Main routine to create and train a Fully Convolutional Network
    for Semantic Segmentation.
    """

    # initialization
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # training hyper parameters
    epochs = 25
    batch_size = 1
    lr = 0.0001
    learning_rate = tf.constant(lr)

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Start training session
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
        vgg_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)
        nn_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(
            sess=sess,
            epochs=epochs,
            batch_size=batch_size,
            get_batches_fn=get_batches_fn,
            train_op=train_op,
            cross_entropy_loss=cross_entropy_loss,
            input_image=vgg_input,
            correct_label=correct_label,
            keep_prob=keep_prob,
            learning_rate=lr,
            runs_dir=runs_dir,
            data_dir=data_dir,
            image_shape=image_shape,
            logits=logits,
            save_epoch=(int)(epochs/5))

        # Save inference data using save_inference_samples
        save_inference_samples(
            runs_dir, data_dir, sess, image_shape,
            logits, keep_prob, vgg_input, 'FINAL')


if __name__ == '__main__':
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), \
        'Please use TensorFlow version 1.0 or newer.' + '  You are using {}'.format(tf.__version__)

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')

    # Unit test all the function implementations
    tests.test_load_vgg(load_vgg, tf)
    tests.test_layers(layers)
    tests.test_optimize(optimize)
    tests.test_train_nn(train_nn)
    tests.test_for_kitti_dataset('./data')

    # Run the FCN
    run()
