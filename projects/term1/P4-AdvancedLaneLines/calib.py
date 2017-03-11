import numpy as np
import pickle
from scipy.misc import imresize, imread
import cv2
from glob import glob
import os


def calibrate_camera(calibration_path, num_corners, image_size):
    """
    This routine first checks if an existing pickle is available and returns it if there is,
    if not, then it calculates the camera calibration using a number of pictures present in the
    calibration_path. It checks for num_corners chess board corners in each of the calibration
    image. It also resizes any calibration image that is not already image_size
    :param calibration_path: Glob path pattern where the calibration images are present
    :param num_corners: Tuple of number of chess board corners in each image (row, col)
    :param image_size: Size of the calibration image (if the original image is not matching this size, then
    the image is resized
    :return: The calibration parameters as a dict{mtx, dist}
    """
    pickle_path = os.path.join(calibration_path,'calibration.p')
    if not os.path.exists(pickle_path):
        glob_pattern = os.path.join(calibration_path, 'calibration*.jpg')
        calibration = __calculate_calibration(calibration_path, num_corners, image_size)
        with open(pickle_path, 'wb') as f:
            pickle.dump(calibration, file=f)
    else:
        with open(pickle_path, "rb") as f:
            calibration = pickle.load(f)

    return calibration


def undistort(img, calibration):
    """
    Takes an image and a calibration object and returns the undistorted image.
    :param img: The image to be undistorted
    :param calibration: The calibration parameters as a dict{mtx, dist}
    :return The undistorted image
    """
    mtx = calibration['mtx']
    dist = calibration['dist']

    result = cv2.undistort(img, mtx, dist, None, mtx)
    return result


def __calculate_calibration(path_pattern, num_corners, image_size):
    """
    Calculates the camera calibration based in chessboard images.
    :param path_pattern: Glob path pattern where the calibration images are present
    :param num_corners: Tuple of number of chess board corners in each image (row, col)
    :param image_size: Size of the calibration image (if the original image is not matching this size, then
    the image is resized
    :return: The calibration parameters as a dict{mtx, dist}
    """
    objp = np.zeros((num_corners[0] * num_corners[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_corners[1], 0:num_corners[0]].T.reshape(-1, 2)

    obj_points = []
    img_points = []

    images = glob(path_pattern)

    successful_cnt = 0
    for idx, fname in enumerate(images):
        img = imread(fname)
        if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
            img = imresize(img, image_size)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (num_corners[1], num_corners[0]), None)

        if ret:
            successful_cnt += 1

            obj_points.append(objp)
            img_points.append(corners)

    print("%s/%s camera calibration images processed." % (successful_cnt, len(images)))

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size[:-1], None, None)
    assert ret, 'Camera calibration failed'

    # Form and return the calibration values
    calibration = {'mtx': mtx,
                   'dist': dist}
    return calibration


