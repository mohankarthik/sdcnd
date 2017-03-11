import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from moviepy.editor import VideoFileClip

"""
CONSTANTS
"""
USE_SOBEL = False
SOBEL_KERNEL = 3
SOBEL_ABS_THRESHOLD = (64, 255)
SOBEL_MAG_THRESHOLD = (128, 255)
SOBEL_DIR_THRESHOLD = (.7, 1.2)

USE_CANNY = False
CANNY_THRESHOLD_LOW = 30
CANNY_THRESHOLD_HIGH = 40

USE_SAT = False
SAT_THRESHOLD_MIN = 128
SAT_THRESHOLD_MAX = 255

USE_WHITE_MASK = True
WHITE_MASK_MIN = (0, 0, 200)
WHITE_MASK_MAX = (255, 30, 255)

USE_YELLOW_MASK = True
YELLOW_MASK_MIN = (15, 100, 100)
YELLOW_MASK_MAX = (35, 255, 255)

USE_GAMMA = True
GAMMA_VALUE = .1
GAMMA_THRESHOLD_MIN = 50
GAMMA_THRESHOLD_MAX = 255

GUASSIAN_KERNEL = 1

PERSPECTIVE_OFFSET = 200
PERSPECTIVE_SRC = np.float32([
    (300, 720),
    (580, 360),
    (730, 360),
    (1100, 720)])
PERSPECTIVE_DST = np.float32([
    (0, 720),
    (0, 0),
    (1, 0),
    (PERSPECTIVE_SRC[-1][0] - PERSPECTIVE_OFFSET, PERSPECTIVE_SRC[0][1])])


def calibrate_camera(calibration_dir, num_corners=(9, 6)):
    """
    Calibrates the camera based on a set of images
    :param calibration_dir: A directory where the calibration images are stored
    :param num_corners: Number of corners in each of the chessboard images
    :return: mtx, dist: The transformation matrix along with the distortion
    """
    # Preconditions
    assert os.path.exists(calibration_dir)

    # Constants
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Initialize the image and object points
    obj_points = []
    img_points = []
    gray_shape = None

    # Setup the object points
    objp = np.zeros((num_corners[0]*num_corners[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners[0],0:num_corners[1]].T.reshape(-1,2)

    # Get the list of images
    img_list = os.listdir(calibration_dir)

    # Loop through the images
    for img_path in img_list:

        # Read the image
        img = cv2.imread(os.path.join(calibration_dir, img_path))

        # Convert to gray-scale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape

        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

        # If corners were found
        if ret:
            # Append the predefined values into obj_pointed
            obj_points.append(objp)

            # Increase the accuracy of the detected corners and add them into img_points
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

    # Let's calibrate the camera with all the known points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray_shape[::-1], None, None)

    # Return the matrix and the distortion
    return mtx, dist


def sobel(img, ksize=SOBEL_KERNEL, abs_thresh=SOBEL_ABS_THRESHOLD,
          mag_thresh=SOBEL_MAG_THRESHOLD, dir_thresh=SOBEL_DIR_THRESHOLD):
    """
    Returns a sobel threshold in the x direction of the image specified by the various parameters
    :param img: The image to be sobel threshold
    :param ksize: Size of the Sobel kernel
    :param abs_thresh: Thresholds for the absolute sobel gradient in the x direction
    :param mag_thresh: Thresholds for the magnitude of sobel gradient
    :param dir_thresh: Thresholds for the direction of sobel gradient
    :return: Threshold image
    """
    # Take the Sobel in both x and y direction
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)

    # Take the magnitude
    abs_sobelx = np.absolute(sobel_x)
    abs_sobely = np.absolute(sobel_y)
    abs_sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    # Compute the direction
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)

    # Scale it to 0 - 255
    scaled_abs_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_mag_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # Create a binary image
    answer = np.zeros_like(scaled_abs_sobelx)
    answer[((scaled_abs_sobelx >= abs_thresh[0]) & (scaled_abs_sobelx <= abs_thresh[1])) |
           ((scaled_mag_sobel >= mag_thresh[0]) & (scaled_mag_sobel <= mag_thresh[1]) &
            (dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1]))] = 255

    # fig = plt.figure()
    # fig.suptitle('Sobel')
    # fig.add_subplot(231), plt.imshow(img, cmap='gray'), plt.title('Sobel Input')
    # fig.add_subplot(232), plt.imshow(scaled_abs_sobelx, cmap='gray'), plt.title('Sobelx')
    # fig.add_subplot(234), plt.imshow(scaled_mag_sobel, cmap='gray'), plt.title('Sobel Magnitude')
    # fig.add_subplot(235), plt.imshow(dir_sobel, cmap='gray'), plt.title('Sobel Direction')
    # fig.add_subplot(236), plt.imshow(answer, cmap='gray'), plt.title('Final')
    # plt.show()

    # Return the binary image
    return answer


def adjust_gamma(image, gamma=GAMMA_VALUE):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def extract_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, YELLOW_MASK_MIN, YELLOW_MASK_MAX)

    return mask


def extract_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, WHITE_MASK_MIN, WHITE_MASK_MAX)

    return mask


def threshold_image(img):
    # Initialize the return
    unified_binary = np.zeros(shape=(6, img.shape[0], img.shape[1]), dtype=np.uint8)

    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Gamma
    if USE_GAMMA:
        gamma = adjust_gamma(img)
        gamma = np.mean(gamma, 2)
        _, unified_binary[0, :, :] = cv2.threshold(gamma.astype(np.uint8), GAMMA_THRESHOLD_MIN, GAMMA_THRESHOLD_MAX, cv2.THRESH_BINARY)

    # Canny
    if USE_CANNY:
        canny = cv2.Canny(l_channel, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)
        canny_binary = np.zeros_like(canny)
        canny_binary[canny > 128] = 255
        unified_binary[1, :, :] = canny_binary

    # Sobel x and threshold the gradient
    if USE_SOBEL:
        sobel_binary = sobel(l_channel)
        unified_binary[2, :, :] = sobel_binary

    # Threshold saturation channel
    if USE_SAT:
        sat_binary = np.zeros_like(s_channel)
        sat_binary[(s_channel >= SAT_THRESHOLD_MIN) & (s_channel <= SAT_THRESHOLD_MAX)] = 255
        unified_binary[3, :, :] = sat_binary

    # White mask
    if USE_WHITE_MASK:
        white_binary = extract_white(img)
        unified_binary[4, :, :] = white_binary

    if USE_YELLOW_MASK:
        yellow_binary = extract_yellow(img)
        unified_binary[5, :, :] = yellow_binary

    # Unified Binary
    answer = np.max(unified_binary, axis=0)

    # Debug
    fig = plt.figure()
    fig.add_subplot(3, 3, 1), plt.imshow(img), plt.title('Original')
    fig.add_subplot(3, 3, 2), plt.imshow(unified_binary[0, :, :], cmap='gray'), plt.title('Gamma')
    fig.add_subplot(3, 3, 3), plt.imshow(unified_binary[1, :, :], cmap='gray'), plt.title('Canny')
    fig.add_subplot(3, 3, 4), plt.imshow(unified_binary[2, :, :], cmap='gray'), plt.title('Sobel')
    fig.add_subplot(3, 3, 5), plt.imshow(unified_binary[3, :, :], cmap='gray'), plt.title('Saturation')
    fig.add_subplot(3, 3, 6), plt.imshow(unified_binary[4, :, :], cmap='gray'), plt.title('White')
    fig.add_subplot(3, 3, 7), plt.imshow(unified_binary[5, :, :], cmap='gray'), plt.title('Yellow')
    fig.add_subplot(3, 3, 8), plt.imshow(answer, cmap='gray'), plt.title('Final')
    plt.show()
    return answer


def histogram(img):
    histogram = np.sum(img[img.shape[0] / 2:, :], axis=0)
    # plt.plot(histogram)
    # plt.show()


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros(shape=img.shape, dtype=np.float32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, np.int32([vertices]), ignore_mask_color)
    mask = np.uint8(mask)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def histogram_lane_detection(img, steps, search_window, h_window, v_window):
    all_x = []
    all_y = []
    masked_img = img[:, search_window[0]:search_window[1]]
    histograms = np.zeros((steps, masked_img.shape[1]))
    pixels_per_step = img.shape[0] // steps

    for i in range(steps):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step
        histogram = np.sum(masked_img[end:start, :], axis=0)
        histograms[i] = histogram

    histograms = histogram_smoothing(histograms, window=v_window)

    for i, histogram in enumerate(histograms):
        start = masked_img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        histogram_smooth = signal.medfilt(histogram, h_window)
        peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 50)))

        highest_peak = detect_highest_peak_in_area(histogram_smooth, peaks, threshold=1000)
        if highest_peak is not None:
            center = (start + end) // 2
            x, y = get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

            all_x.extend(x)
            all_y.extend(y)

    all_x = np.array(all_x) + search_window[0]
    all_y = np.array(all_y)

    return all_x, all_y


def highest_n_peaks(histogram, peaks, n=2, threshold=0):
    if len(peaks) == 0:
        return []

    peak_list = []
    for peak in peaks:
        y = histogram[peak]
        if y > threshold:
            peak_list.append((peak, histogram[peak]))
    peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

    if len(peak_list) == 0:
        return []
    else:
        x, y = zip(*peak_list)
        return x[:n]


def histogram_smoothing(histograms, window=3):
    smoothed = np.zeros_like(histograms)
    for h_i, hist in enumerate(histograms):
        window_sum = np.zeros_like(hist)
        for w_i in range(window):
            index = w_i + h_i - window // 2
            if index < 0:
                index = 0
            elif index > len(histograms) - 1:
                index = len(histograms) - 1

            window_sum += histograms[index]

        smoothed[h_i] = window_sum / window

    return smoothed


def detect_highest_peak_in_area(histogram, peaks, threshold=0):
    peak = highest_n_peaks(histogram, peaks, n=1, threshold=threshold)
    if len(peak) == 1:
        return peak[0]
    else:
        return None


def detect_lane_along_poly(img, poly, steps):
    pixels_per_step = img.shape[0] // steps
    all_x = []
    all_y = []

    for i in range(steps):
        start = img.shape[0] - (i * pixels_per_step)
        end = start - pixels_per_step

        center = (start + end) // 2
        x = poly(center)

        x, y = get_pixel_in_window(img, x, center, pixels_per_step)

        all_x.extend(x)
        all_y.extend(y)

    return all_x, all_y


def get_pixel_in_window(img, x_center, y_center, size):
    half_size = size // 2
    window = img[y_center - half_size:y_center + half_size,
             x_center - half_size:x_center + half_size]

    x, y = (window.T == 255).nonzero()

    x = x + x_center - half_size
    y = y + y_center - half_size

    return x, y


def calculate_lane_area(lanes, img_height, steps):
    """
    Expects the line polynom to be a function of y.
    """
    points_left = np.zeros((steps + 1, 2))
    points_right = np.zeros((steps + 1, 2))

    for i in range(steps + 1):
        pixels_per_step = img_height // steps
        start = img_height - i * pixels_per_step

        points_left[i] = [lanes[0].best_fit_poly(start), start]
        points_right[i] = [lanes[1].best_fit_poly(start), start]

    return np.concatenate((points_left, points_right[::-1]), axis=0)


def are_lanes_plausible(lane_one, lane_two, parall_thres=(0.0002, 0.5), dist_thres=(450, 550)):
    is_parall = lane_one.is_current_fit_parallel(lane_two, threshold=parall_thres)
    dist = lane_one.get_current_fit_distance(lane_two)
    is_plausible_dist = dist_thres[0] < dist < dist_thres[1]
    return is_parall & is_plausible_dist


def draw_poly(img, poly, steps, color, thickness=10, dashed=False):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.line(img, end_point, start_point, color, thickness)

    return img


def draw_poly_arr(img, poly, steps, color, thickness=10, dashed=False, tipLength=1):
    img_height = img.shape[0]
    pixels_per_step = img_height // steps

    for i in range(steps):
        start = i * pixels_per_step
        end = start + pixels_per_step

        start_point = (int(poly(start)), start)
        end_point = (int(poly(end)), end)

        if dashed == False or i % 2 == 1:
            img = cv2.arrowedLine(img, end_point, start_point, color, thickness, tipLength=tipLength)

    return img


def outlier_removal(x, y, q=10):
    x = np.array(x)
    y = np.array(y)

    lower_bound = np.percentile(x, q)
    upper_bound = np.percentile(x, 100 - q)
    selection = (x >= lower_bound) & (x <= upper_bound)
    return x[selection], y[selection]


def calc_curvature(fit_cr):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension

    y = np.array(np.linspace(0, 719, num=10))
    x = np.array([fit_cr(x) for x in y])
    y_eval = np.max(y)

    fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad

class PerspectiveTransform:
    """
    Transforms an image with perspective transform
    """
    def __init__(self, src, dst):
        """
        Few source and destination points to initialize the transform
        :param src: Source points
        :param dst: Desination points
        """
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """
        Transforms the image moving from a perspective view to an overhead view
        :param img: Image to be transformed
        :return: Transformed image
        """
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        """
        Performs inverse transform on the image going from overhead view to perspective view
        :param img: Image to be transformed
        :return: Transformed image
        """
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)



class LaneDetector:

    def __init__(self, mtx, dist, img_shape):
        print (img_shape)
        self.mtx = mtx
        self.dist = dist
        # self.perspective_src = np.float32([
        #     (img_shape[1] * 0., img_shape[0] * .9),
        #     (img_shape[1] * .5, img_shape[0] * .5),
        #     (img_shape[1] * .5, img_shape[0] * .5),
        #     (img_shape[1] * 1., img_shape[0] * .9)])
        # self.perspective_dst = np.float32([
        #     (100, img_shape[0] - 100),
        #     (100, 100),
        #     (img_shape[1] - 100, 100),
        #     (img_shape[1] - 100, img_shape[0] - 100)])
        self.perspective_src = np.float32([
            (img_shape[1] * 0.10, img_shape[0] * 0.9),
            (img_shape[1] * 0.46, img_shape[0] * 0.6),
            (img_shape[1] * 0.54, img_shape[0] * 0.6),
            (img_shape[1] * 0.90, img_shape[0] * 0.9)])
        self.perspective_dst = np.float32([
            (img_shape[1] * 0.0, img_shape[0] * 1.0),
            (img_shape[1] * 0.0, img_shape[0] * 0.0),
            (img_shape[1] * 1.0, img_shape[0] * 0.0),
            (img_shape[1] * 1.0, img_shape[0] * 1.0)])
        self.ROI = np.array([
                    [img_shape[1] * 0.10, img_shape[0] * 0.9],
                    [img_shape[1] * 0.46, img_shape[0] * 0.6],
                    [img_shape[1] * 0.54, img_shape[0] * 0.6],
                    [img_shape[1] * 0.90, img_shape[0] * 0.9]])
        self.perspective_transformer = PerspectiveTransform(self.perspective_src, self.perspective_dst)

    def __line_found(self, left, right):
        if len(left[0]) == 0 or len(right[0]) == 0:
            return False
        else:
            left_x, left_y = outlier_removal(left[0], left[1])
            right_x, right_y = outlier_removal(right[0], right[1])
            new_left = Line(y=left_x, x=left_y)
            new_right = Line(y=right_x, x=right_y)
            return are_lanes_plausible(new_left, new_right)
        
    def process(self, img):
        """
        Processes a single image
        :param img: The color image to be processed (should be in RBG)
        :return: The processes image with the lane lines drawn on it
        """
        # Make a copy
        img = np.copy(img)

        # Undistort the image
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

        # Aply a guassian blur
        blur = cv2.GaussianBlur(undist, (GUASSIAN_KERNEL, GUASSIAN_KERNEL), 0)

        # Threshold the image
        thresh = threshold_image(blur)

        # Apply ROI
        crop = region_of_interest(img, self.ROI)

        # Transform the image
        fig = plt.figure()
        fig.add_subplot(121), plt.imshow(crop, cmap='gray')
        birdseye = self.perspective_transformer.transform(thresh)
        fig.add_subplot(122), plt.imshow(birdseye, cmap='gray')
        plt.show()

        left_x, left_y = histogram_lane_detection(
            birdseye, 10, (0, birdseye.shape[1] // 2), h_window=21, v_window=3)
        right_x, right_y = histogram_lane_detection(
            birdseye, 10, (birdseye.shape[1] // 2, birdseye.shape[1] - 0), h_window=21, v_window=3)

        lines_detected = self.__line_found((left_x, left_y), (right_x, right_y))

        if lines_detected == True or self.n_frames_processed == 0:
            # switch x and y since lines are almost vertical
            self.left_line.update(y=left_x, x=left_y)
            # switch x and y since lines are almost vertical
            self.right_line.update(y=right_x, x=right_y)

            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = calc_curvature(self.center_poly)
            self.offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700
            #         except:
            #             print('error while searching a line')

        self.__draw_lane_overlay(orig_frame)
        self.__draw_info_panel(orig_frame)

        self.n_frames_processed += 1


if __name__ == "__main__":
    mtx, dist = calibrate_camera('camera_cal')

    # yellow_output = 'output.mp4'
    # clip2 = VideoFileClip('VID_20161210_181736.mp4')
    # yellow_clip = clip2.fl_image(ld.process)
    # yellow_clip.write_videofile(yellow_output, audio=False)

    test_imgs = os.listdir('test_images')

    for img_path in test_imgs:
        if img_path.find('test') is not -1:
            print (img_path)
            img = cv2.imread(os.path.join('test_images', img_path))
            ld = LaneDetector(mtx, dist, img.shape)
            ld.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
