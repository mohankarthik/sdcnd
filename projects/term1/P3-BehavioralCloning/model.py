# Import generic utils
import os

# Import math libs
import numpy as np
import pandas as pd

# Import image processing libs
import cv2

# Import the keras layers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, Lambda
from keras.models import model_from_json
from keras.optimizers import Adam

"""
CONSTANTS
"""
PATH = '/home/karti/sdc-live-trainer/data'  # Data path

# Data augmentation constants
TRANS_X_RANGE = 100  # Number of translation pixels up to in the X direction for augmented data (-RANGE/2, RANGE/2)
TRANS_Y_RANGE = 40  # Number of translation pixels up to in the Y direction for augmented data (-RANGE/2, RANGE/2)
TRANS_ANGLE = .3  # Maximum angle change when translating in the X direction
OFF_CENTER_IMG = .25  # Angle change when using off center images

BRIGHTNESS_RANGE = .25  # The range of brightness changes
ANGLE_THRESHOLD = 1.  # The maximum magitude of the angle possible

# Training constants
BATCH = 128  # Number of images per batch
TRAIN_BATCH_PER_EPOCH = 160  # Number of batches per epoch for training
TRAIN_VAL_CHECK = 1e-3  # The maximum increase in validation loss during re-training
EPOCHS = 10  # Minimum number of epochs to train the model on

# Image constants
IMG_ROWS = 64  # Number of rows in the image
IMG_COLS = 64  # Number of cols in the image
IMG_CH = 3  # Number of channels in the image


def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    # Remove the unwanted top scene and retain only the track
    roi = img[60:140, :, :]

    # Resize the image
    resize = cv2.resize(roi, (IMG_ROWS, IMG_COLS), interpolation=cv2.INTER_AREA)

    # Return the image sized as a 4D array
    return np.resize(resize, (1, IMG_ROWS, IMG_COLS, IMG_CH))


def img_change_brightness(img):
    # Convert the image to HSV
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Compute a random brightness value and apply to the image
    brightness = BRIGHTNESS_RANGE + np.random.uniform()
    temp[:, :, 2] = temp[:, :, 2] * brightness

    # Convert back to RGB and return
    return cv2.cvtColor(temp, cv2.COLOR_HSV2RGB)


def img_translate(img, x_translation):
    # Randomly compute a Y translation
    y_translation = (TRANS_Y_RANGE * np.random.uniform()) - (TRANS_Y_RANGE / 2)

    # Form the translation matrix
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

    # Translate the image
    return cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))


def data_augment(img_path, angle, threshold, bias):
    """
    Augments the data by generating new images based on the base image found in img_path
    :param img_path: Path to the image to be used as the base image
    :param angle: The steering angle of the current image
    :param threshold: If the new angle is below this threshold, then the image is dropped
    :return:
        new_img, new_angle of the augmented image / angle (or)
        None, None if the new angle is below the threshold
    """
    # Randomly form the X translation distance and compute the resulting steering angle change
    x_translation = (TRANS_X_RANGE * np.random.uniform()) - (TRANS_X_RANGE / 2)
    new_angle = angle + ((x_translation / TRANS_X_RANGE) * 2) * TRANS_ANGLE

    # Check if the new angle does not meets the threshold requirements
    if (abs(new_angle) + bias) < threshold or abs(new_angle) > 1.:
        return None, None

    # Let's read the image
    img = cv2.imread(img_path)  # Read in the image
    img = img_change_brightness(img)  # Randomly change the brightness
    img = img_translate(img, x_translation)  # Translate the image in X and Y direction
    if np.random.randint(2) == 0:  # Flip the image
        img = np.fliplr(img)
        new_angle = -new_angle
    img = img_pre_process(img)  # Pre process the image

    return img, new_angle


def val_data_generator(df):
    """
    Validation data generator
    :param df: Pandas data frame consisting of all the validation data
    :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)
    """
    # Preconditions
    assert len(df) == BATCH, 'The length of the validation set should be batch size'

    while 1:
        _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
        _y = np.zeros(BATCH, dtype=np.float)

        for idx in np.arange(BATCH):
            _x[idx] = img_pre_process(cv2.imread(os.path.join(PATH, df.center.iloc[idx].strip())))
            _y[idx] = df.steering.iloc[idx]

        yield _x, _y


def train_data_generator(df, bias):
    """
    Training data generator
    :param df: Pandas data frame consisting of all the training data
    :return: (x[BATCH, IMG_ROWS, IMG_COLS, NUM_CH], y)
    """
    _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
    _y = np.zeros(BATCH, dtype=np.float)
    out_idx = 0
    while 1:
        # Get a random line and get the steering angle
        idx = np.random.randint(len(df))
        angle = df.steering.iloc[idx]

        """
        ANGLE SMOOTHING
        ---------------
        The human driven angle data is generally quite extreme. Empirically,
         softening the angles by dividing them by a constant and making them
         smoother allows the car to drive around the track much more smoothly.

        But note, this counter acts our idea that we want to bias the model
        towards bigger angles and not let it bias to 0. So we need to walk a
        fine line and this is a step to be taken near the end of the training
        session
        """
        # Pick one of the images, left, right or center
        img_choice = np.random.randint(3)

        if img_choice == 0:
            img_path = os.path.join(PATH, df.left.iloc[idx].strip())
            angle += OFF_CENTER_IMG
        elif img_choice == 1:
            img_path = os.path.join(PATH, df.center.iloc[idx].strip())
        else:
            img_path = os.path.join(PATH, df.right.iloc[idx].strip())
            angle -= OFF_CENTER_IMG

        """
        Randomly distort the (img, angle) to generate new data
        Here, we want to bias towards not selecting low angles, so we generate a random number
        and if that number were less than the absolute value of the newly coined angle + a known bias,
        only then do we accept the transformation.
        """
        threshold = np.random.uniform()
        img, angle = data_augment(img_path, angle, threshold, bias)

        # Check if we've got valid values
        if img is not None:
            _x[out_idx] = img
            _y[out_idx] = angle
            out_idx += 1

        # Check if we've enough values to yield
        if out_idx >= BATCH:
            yield _x, _y

            # Reset the values back
            _x = np.zeros((BATCH, IMG_ROWS, IMG_COLS, IMG_CH), dtype=np.float)
            _y = np.zeros(BATCH, dtype=np.float)
            out_idx = 0


def get_model():
    """
    Defines the model
    :return: Returns the model
    """
    """
    Check if a model already exists
    """
    if os.path.exists(os.path.join('.', 'model.json')):
        ch = input('A model already exists, do you want to reuse? (y/n): ')
        if ch == 'y' or ch == 'Y':
            with open(os.path.join('.', 'model.json'), 'r') as in_file:
                json_model = in_file.read()
                model = model_from_json(json_model)

            weights_file = os.path.join('.', 'model.h5')
            model.load_weights(weights_file)
            print('Model fetched from the disk')
            model.summary()
            return model

    """
    Reform the VGG16 net
    """
    model = Sequential()

    # Add a normalization layer
    model.add(Lambda(lambda x: x/127.5 - .5,
                     input_shape=(IMG_ROWS, IMG_COLS, IMG_CH),
                     output_shape=(IMG_ROWS, IMG_COLS, IMG_CH)))

    # Add a color map layer as suggested by Vivek Yadav to let the model figure out
    # the best color map for this hypothesis
    model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))

    # Add the VGG like layers
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv1'))
    model.add(Convolution2D(64, 3, 3, activation='elu', border_mode='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv1'))
    model.add(Convolution2D(128, 3, 3, activation='elu', border_mode='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv1'))
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv2'))
    model.add(Convolution2D(256, 3, 3, activation='elu', border_mode='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv1'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv2'))
    model.add(Convolution2D(512, 3, 3, activation='elu', border_mode='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.add(Flatten(name='Flatten'))
    model.add(Dense(1024, activation='elu', name='fc1'))
    model.add(Dropout(0.5, name='fc1_dropout'))
    model.add(Dense(256, activation='elu', name='fc2'))
    model.add(Dropout(0.5, name='fc2_dropout'))
    model.add(Dense(128, activation='elu', name='fc3'))
    model.add(Dropout(0.5, name='fc3_dropout'))
    model.add(Dense(64, activation='elu', name='fc4'))
    model.add(Dropout(0.5, name='fc4_dropout'))
    model.add(Dense(32, activation='elu', name='fc5'))
    model.add(Dropout(0.5, name='fc5_dropout'))
    model.add(Dense(1, init='zero', name='output'))

    """
    Load the VGG16 weights
    """
    model.load_weights('vgg16_weights.h5', by_name=True)

    """
    Print the summary
    """
    model.summary()
    return model


def train_model(model, train_data, val_data):
    """
    Trains the given model
    :param model: A keras model
    :param train_data: Training data as a pandas data frame
    :param val_data: The validation data as a pandas data frame
    :return: The history of the model
    """

    """
    Now that the fully connected layer is fully settled, let's get the full training
    started.
    Note:
    1. We allow more of the VGG16 convnet to be fine-tuned
    2. When we are retraining, we'll start directly from here
    3. we start with a smaller learning rate so as to not over-fit the data
    4. We enable data augmentation so that the data generalizes now
    """
    # Make the top 2 and the bottom two Conv Layers along with all the FC layers trainable
    for layer in model.layers[0:2]:
        layer.trainable = True
    for layer in model.layers[2:12]:
        layer.trainable = False
    for layer in model.layers[12:]:
        layer.trainable = True

    # Recompile the model with a finer learning rate
    model.compile(optimizer=Adam(1e-5), loss='mse')

    # Get an evaluation on the validation set
    val_loss = model.evaluate_generator(val_data_generator(val_data), val_samples=BATCH)
    print('Pre-trained evaluation loss = {}'.format(val_loss))

    # Try some predictions before we start..
    test_predictions(model, train_data)

    num_runs = 0
    while True:
        bias = 1. / (num_runs + 1.)

        print('Run {} with bias {}'.format(num_runs+1, bias), end=': ')

        history = model.fit_generator(
            train_data_generator(train_data, bias),
            samples_per_epoch=TRAIN_BATCH_PER_EPOCH * BATCH,
            nb_epoch=1,
            validation_data=val_data_generator(val_data),
            nb_val_samples=BATCH,
            verbose=1)
        num_runs += 1

        # Print out the test predictions
        test_predictions(model, train_data)

        # Save the model and the weights so far as checkpoints so we can manually terminate when things
        # go south...
        # Think that statement is very offensive to the south though, let's call it
        # when things go north :P
        save_model(model, num_runs)

        # If the validation loss starts to increase, it's time for us to stop training...
        if num_runs > EPOCHS:
            break


def test_predictions(model, df, num_tries=5):
    """
    Tries some random predictions
    :param model: The keras model
    :param df: The validation data as a pandas data frame
    :param num_tries: Number of images to try on
    :return: None
    """
    print('Predictions: ')
    for i in np.arange(num_tries):
        topset = df.loc[df.steering < (i * .4) - .6]
        subset = topset.loc[topset.steering >= (i * .4) - 1.]
        idx = int(len(subset)/2)
        img = img_pre_process(cv2.imread(os.path.join(PATH, subset.center.iloc[idx].strip())))
        img = np.resize(img, (1, IMG_ROWS, IMG_COLS, IMG_CH))
        org_angle = subset.steering.iloc[idx]
        pred_angle = model.predict(img, batch_size=1)
        print(org_angle, pred_angle[0][0])


def save_model(model, epoch=''):
    """
    Saves the model and the weights to a json file
    :param model: The mode to be saved
    :param epoch: The epoch number, so as to save the model to a different file name after each epoch
    :return: None
    """
    json_string = model.to_json()
    with open('model'+str(epoch)+'.json', 'w') as outfile:
        outfile.write(json_string)
    model.save_weights('model'+str(epoch)+'.h5')
    print('Model saved')


if __name__ == '__main__':
    # Set the seed for predictability
    np.random.seed(200)

    # Load the data
    total_data = pd.read_csv(os.path.join(PATH, 'driving_log.csv'))

    # Shuffle and split the data set
    validate, train = np.split(total_data.sample(frac=1), [BATCH])
    del total_data

    # Create a model
    steering_model = get_model()

    # Train the model
    train_model(steering_model, train, validate)

    exit(0)
