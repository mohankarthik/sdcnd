import os
import glob
import pickle
from utils import *
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def process_features():
    """
    Processes the features and saves the features in a pickle file
    """
    # ----- PARAMETERS ----- #
    parameters = {'color_space': 'HSV',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                  'hog_orientations': 8,  # Number of orientations for HOG features
                  'hog_pix_per_cell': 8,  # Number of pixels per cell in HOG features
                  'hog_cell_per_block': 2,  # Number of cells per block in HOG features
                  'hog_channel': [1, 2],  # HOG features to be extracted from which channel
                  'spatial_size': (16, 16),  # Size of the spatial features
                  'color_hist_bins': 16,  # Number of color histogram bins
                  'spatial_enabled': True,  # If spatial features should be included
                  'color_hist_enabled': True,  # If color features should be included
                  'hog_enabled': True  # If HOG features should be included
                  }

    # Read the image paths
    cars = []
    not_cars = []

    cars_images = glob.glob('data/vehicles/*')
    for folder in cars_images:
        cars += glob.glob('{}/*.png'.format(folder))

    not_cars_images = glob.glob('data/non-vehicles/*')
    for folder in not_cars_images:
        not_cars += glob.glob('{}/*.png'.format(folder))

    print("Number of car samples:", len(cars))
    print("Number of non-car samples", len(not_cars))

    # Extract the features for both the sets of data
    car_features = extract_features(cars, parameters)
    not_car_features = extract_features(not_cars, parameters)

    # Stack them together
    features = np.vstack((car_features, not_car_features)).astype(np.float64)

    # Scale and normalize the features
    scaler = StandardScaler().fit(features)
    scaled_features = scaler.transform(features)

    # Define the labels vector
    targets = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))

    # Split up data into randomized training and test sets
    features_train, features_test, targets_train, targets_test = train_test_split(
        scaled_features, targets, test_size=0.2, random_state=42)

    print('Feature vector length:', len(features_train[0]))

    # Save the data for easy access
    pickle_file = 'features.pickle'
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'features_train': features_train,
                    'targets_train': targets_train,
                    'features_test': features_test,
                    'targets_test': targets_test,
                    'scaler': scaler,
                    'parameters': parameters
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


def train_classifier():
    # Reload the data
    pickle_file = 'features.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        features_train = pickle_data['features_train']
        targets_train = pickle_data['targets_train']
        features_test = pickle_data['features_test']
        targets_test = pickle_data['targets_test']
        scaler = pickle_data['scaler']
        parameters = pickle_data['parameters']
        del pickle_data  # Free up memory

    print('Data and modules loaded.')
    print("train_features size:", features_train.shape)
    print("train_labels size:", targets_train.shape)
    print("test_features size:", features_test.shape)
    print("test_labels size:", targets_test.shape)
    for k in parameters:
        print(k, ":", parameters[k])

    # Use a linear SVC
    svc = LinearSVC(max_iter=20000)
    svc.fit(features_train, targets_train)

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(features_test, targets_test), 4))

    # Save the model for easy access
    pickle_file = 'svc.pickle'
    print('Saving data to pickle file...')
    try:
        with open(pickle_file, 'wb') as pfile:
            pickle.dump(
                {
                    'svc': svc,
                    'scaler': scaler,
                    'parameters': parameters
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise


if __name__ == "__main__":
    # Check if a pickle file already exists
    if os.path.exists('features.pickle'):
        char = input('A features pickle already exists, do you want to reuse (y/n):')
        if char == 'N' or char == 'n':
            process_features()

    # Check if a classifier picklt already exists
    if os.path.exists('svc.pickle'):
        char = input('A classified pickle already exists, do you want to reuse (y/n):')
        if char == 'N' or char == 'n':
            train_classifier()
