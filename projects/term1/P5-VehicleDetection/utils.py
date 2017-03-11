import matplotlib.image as mpimg
import numpy as np
import cv2
import copy
from scipy import ndimage as ndi
from skimage.feature import hog


# ------------- FEATURE EXTRACTOR ------------- #
def extract_features(img_paths, parameters):
    """
    Core function of utils. Extracts a combination of features from a image

    :param img_paths: A list of image paths
    :param parameters: A dictionary of parameters

    :return: A concatenated set of features
    """
    features = []
    for file in img_paths:
        image = mpimg.imread(file)
        if np.max(image) > 1:
            image = image.astype(np.float32) / float(np.max(image))

        # Extract the features of each image
        features.append(_extract_img_features(image, parameters))

    return features


def get_windows(img, x_range=(None, None), y_range=(None, None),
                window_size=(64, 64), overlap=(0.5, 0.5)):
    """
    Takes an image, range of x and y, and computes all possible windows over the image with
    a defined overlap

    :param img: The image over which to form the sliding windows
    :param x_range: The range of x values for the windows to span as a tuple
    :param y_range: The range of y values for the windows to span as a tuple
    :param window_size: The size of the window as a tuple
    :param overlap: The overlap of the window in x and y direction as a tuple
    :return:
    """
    # If the X or Y range is a None, set it to the min / max size of the image
    x_range = (
        x_range[0] if x_range[0] is not None else 0,
        x_range[1] if x_range[1] is not None else img.shape[1],
    )
    y_range = (
        y_range[0] if y_range[0] is not None else 0,
        y_range[1] if y_range[1] is not None else img.shape[0],
    )

    # Compute the span of the region to be searched
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(window_size[0] * (1 - overlap[0]))
    ny_pix_per_step = np.int(window_size[1] * (1 - overlap[1]))

    # Compute the number of windows in x/y
    nx_windows = np.int(x_span / nx_pix_per_step) - 1
    ny_windows = np.int(y_span / ny_pix_per_step) - 1

    # Initialize a list to append window positions to
    window_list = []

    # Find the windows
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            x_start = xs * nx_pix_per_step + x_range[0]
            x_end = x_start + window_size[0]
            y_start = ys * ny_pix_per_step + y_range[0]
            y_end = y_start + window_size[1]

            # Append window position to list
            window_list.append(((x_start, y_start), (x_end, y_end)))

    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bounding_boxes, color=(255, 0, 0), thick=6):
    """
    Draws boxes on an image given the image and the bounding boxes

    :param img: The image to be drawn on
    :param bounding_boxes: Specification of the boxes [[(x1, y1), (x2, y2)], ...]
    :param color: The color of the box (by default red)
    :param thick: Thickness of the box (by default 6)
    :return: Returns the image with boxes drawn
    """
    imcopy = np.copy(img)
    for bbox in bounding_boxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def find_cars(img, windows, clf, feature_scaler, parameters):
    """
    Searches an image using windows for possible car locations

    :param img: The image to search on
    :param windows: Set of windows to use while searching the image
    :param clf: The sklearn classifier to use to search the image
    :param feature_scaler: Feature scaler used to scale the images during training
    :param parameters: A dictionary of parameters
    :return: windows where the cars were found
    """
    result = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = _extract_img_features(test_img, parameters)

        test_features = feature_scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        conf = clf.decision_function(test_features)
        if prediction == 1 and conf > 0.4:
            result.append(window)

    return result


def create_heatmap(windows, image_shape):
    """
    Creates a heatmap by overlapping windows

    :param windows: A list of windows
    :param image_shape: Shape of the image
    :return: Returns a heatmap image
    """
    heatmap = np.zeros(image_shape[:2])
    for window in windows:
        heatmap[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return heatmap


def combine_boxes(windows, image_shape, threshold=0):
    hot_windows = []

    # If there are any windows
    if len(windows) > 0:
        image = create_heatmap(windows, image_shape)
        hot_windows = find_windows_from_heatmap(image, threshold)

    return hot_windows


def _get_hog_features(img, orient, pix_per_cell, cell_per_block, visualize=False, feature_vec=True):
    """
    Compute HOG features and visualization (if required)

    :param img: The input image
    :param orient: The number of orientations to consider
    :param pix_per_cell: Number of pixels per cell
    :param cell_per_block: Number of cells per block
    :param visualize: If visualization is needed
    :param feature_vec: If the features should be got as a vector

    :return: Returns the features (as a vector if feature_vec=True) and the hog visualization if (vis=True)
    """
    if visualize:
        # Call with two outputs if visualize==True
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=visualize, feature_vector=feature_vec)
        return features, hog_image
    else:
        # Otherwise call with one output
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=visualize, feature_vector=feature_vec)
        return features


def _get_spatial_features(img, size=(32, 32)):
    """
    Compute binned color features

    :param img: The input image
    :param size: Size of the image to be considered (the image is resized to this before binning)

    :return: Returns the spatial binned features
    """
    features = cv2.resize(img, size).ravel()
    return features


def _get_color_features(img, nbins=32, bins_range=(0, 256)):
    """
    Compute color histogram features
    Note: The input image should be of uint8 and of size 0 to 256. If not,
    then please pass an appropriate bins_range

    :param img: The original iamge
    :param nbins: Number of bins of histogram
    :param bins_range: Range of each bin (typically 0 - 256)

    :return: The color histogram of each channel
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features


def _extract_img_features(img, parameters):
    """
    Extracts the features from a single image

    :param img: The image to be processed
    :param parameters: A dictionary of parameters
    :return: Returns a concatenated features
    """
    img_features = []

    # apply color conversion if other than 'RGB'
    feature_image = np.copy(img)
    if parameters['color_space'] != 'RGB':
        if parameters['color_space'] == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif parameters['color_space'] == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif parameters['color_space'] == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif parameters['color_space'] == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif parameters['color_space'] == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    if parameters['spatial_enabled']:
        # Compute the spatial features if it's required
        img_features.append(_get_spatial_features(feature_image, size=parameters['spatial_size']))
    if parameters['color_hist_enabled']:
        # Compute the color histogram features if required
        img_features.append(_get_color_features(feature_image, nbins=parameters['color_hist_bins']))
    if parameters['hog_enabled']:
        # Compute the HOG features if requiread
        hog_features = []
        for channel in parameters['hog_channel']:
            hog_features.extend(_get_hog_features(feature_image[:, :, channel], parameters['hog_orientations'],
                                                  parameters['hog_pix_per_cell'], parameters['hog_cell_per_block'],
                                                  visualize=False, feature_vec=True))
        img_features.append(hog_features)

    return np.concatenate(img_features)


def find_windows_from_heatmap(image, threshold=0):
    """
    Finds a unified window from a heatmap

    :param image: The heatmap image
    :param threshold: The threshold to be used
    :return: Returns a list of windows
    """
    hot_windows = []

    # Threshold the image
    image[image <= threshold] = 0

    # Create labels
    labels = ndi.label(image)

    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        non_zero = (labels[0] == car_number).nonzero()

        # Identify x and y values of those pixels
        non_zero_y = np.array(non_zero[0])
        non_zero_x = np.array(non_zero[1])

        # Define a bounding box based on min/max x and y
        hot_windows.append(((np.min(non_zero_x), np.min(non_zero_y)), (np.max(non_zero_x), np.max(non_zero_y))))

    return hot_windows
