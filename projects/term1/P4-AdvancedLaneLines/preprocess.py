import numpy as np
import cv2


class PerspectiveTransformer:
    def __init__(self, src, dst):
        """
        Initializes the perspective transform with a set of source and destination points

        :param src: Source points
        :param dst: Desination points
        """
        self.src = src
        self.dst = dst
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def transform(self, img):
        """
        Transforms an image from source to destination

        :param img: Image to be transformed
        :return: Transformed images
        """
        return cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    def inverse_transform(self, img):
        """
        Transforms an image from destination to source
        :param img: Image to be transformed
        :return: Transformed image
        """
        return cv2.warpPerspective(img, self.M_inv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


class Sobel:
    """
    Static class to implement all of Sobel methods
    """
    @staticmethod
    def absolute_thresh(img, orientation='x', sobel_kernel=3, threshold=(0, 255)):
        """
        Computes the absolute threshold of an image, either in the x or in the y direction

        :param img: The image to be processed
        :param orientation: Orientation of the sobel processing ('x' or 'y'), by default it's 'x'
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The threshold for the resulting image, by default, it's the entire threshold value
        :return: Image after threshold computation
        """
        if orientation == 'x':
            axis = (1, 0)
        elif orientation == 'y':
            axis = (0, 1)
        else:
            raise ValueError('orientation has to be "x" or "y" not "%s"' % orientation)

        sobel = cv2.Sobel(img, cv2.CV_64F, *axis, ksize=sobel_kernel)
        sobel = np.absolute(sobel)

        scale_factor = np.max(sobel) / 255
        sobel = (sobel / scale_factor).astype(np.uint8)

        result = np.zeros_like(sobel)
        result[(sobel > threshold[0]) & (sobel <= threshold[1])] = 1

        return result

    @staticmethod
    def magnitude_thresh(img, sobel_kernel=3, threshold=(0, 255)):
        """
        Computes the magnitude threshold of an image

        :param img: The image to be processed
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The magnitude threshold for the resulting image
        :return: Image after threshold computation
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        scale_factor = np.max(magnitude) / 255
        magnitude = (magnitude / scale_factor).astype(np.uint8)

        result = np.zeros_like(magnitude)
        result[(magnitude > threshold[0]) & (magnitude <= threshold[1])] = 1

        return result

    @staticmethod
    def direction_threshold(img, sobel_kernel=3, threshold=(0, np.pi / 2)):
        """
        Computes the direction threshold of an image

        :param img: The image to be processed
        :param sobel_kernel: The size of the sobel kernel. Larger the size, the result is more smoothened. By default
        it's 3.
        :param threshold: The direction threshold for the resulting image
        :return: Image after threshold computation
        """
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        with np.errstate(divide='ignore', invalid='ignore'):
            direction = np.absolute(np.arctan(sobel_y / sobel_x))
            result = np.zeros_like(direction)
            result[(direction > threshold[0]) & (direction <= threshold[1])] = 1

        return result


def gamma_threshold(img, gamma=1.0, threshold=(100, 255)):
    """
    build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values

    :param img: Image to be processed
    :param gamma: Gamma correction value for the original image (in range of 0.0 to 1.0)
    :param threshold: Treshold to apply for gamma correction
    :return: Returns the gamma thresholded image
    """
    #
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    result = cv2.LUT(img, table)

    # Threshold the image
    result = np.mean(result, 2)
    _, result = cv2.threshold(result.astype(np.uint8), threshold[0], threshold[1], cv2.THRESH_BINARY)

    return result


def guassian_blur(img, ksize):
    """
    Applies guassian blur on the image of the size specified by ksize and returns

    :param img: Image to be filtered
    :param ksize: Size of the kernel
    :return: Filtered image
    """
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def median_blur(img, ksize):
    """
    Applies median blur on the image of the size specified by ksize and returns

    :param img: Image to be filtered
    :param ksize: Size of the kernel
    :return: Filtered image
    """
    return cv2.medianBlur(img, ksize)

def canny(img, low, high):
    """
    Performs and returns the canny thresholded image

    :param img: The image to be thresholded
    :param low: Low threshold
    :param high: High threshold
    :return: The canny thresholded image
    """
    return cv2.Canny(img, low, high)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, horizon_y, limits=(-0.4, 0.4)):
    """
    Finds an approximate lane using hough transform and returns the x and y co-ordinate pairs

    :param img: The canny thresholded image
    :param rho: Rho resolution
    :param theta: The thetha resolution
    :param threshold: The hough voting threshold
    :param min_line_len: The houghlinesP value for minimum length beyond which a segment is considered as a line
    :param max_line_gap: The houghlinesP value for maximum gap between two lines beyond which they are considered as
    a seperate line
    :param horizon_y: The horizon value
    :param limits: The limits of the slope of the lane
    :return: An numpy array of shape (4,2) that contains the 4 bounding boxes for the lanes, and the left and right
    confidences
    """
    result = np.zeros(shape=(4,2), dtype=np.int32)

    # Find the lines
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # Consolidate the lines into two straight lanes
    if lines is not None:
        left_m = left_b = right_m = right_b = 0
        left_num = right_num = 1e-5

        # Loop through all the lines and find the slopes and intercepts
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    intercept = y2 - (slope * x2)
                    if slope < limits[0]:
                        left_m += slope
                        left_b += intercept
                        left_num += 1
                    elif slope > limits[1]:
                        right_m += slope
                        right_b += intercept
                        right_num += 1

        (m1, b1) = (left_m / left_num, left_b / left_num)
        (m2, b2) = (right_m / right_num, right_b / right_num)

        if m1 and m1 != float('Inf') and b1 and b1 != float('Inf'):
            result[0][0] = (img.shape[0] - b1) / m1
            result[0][1] = img.shape[0]

            result[1][0] = (horizon_y - b1) / m1
            result[1][1] = horizon_y

        if m2 and m2 != float('Inf') and b2 and b2 != float('Inf'):
            result[2][0] = (horizon_y - b2) / m2
            result[2][1] = horizon_y

            result[3][0] = (img.shape[0] - b2) / m2
            result[3][1] = img.shape[0]

        return result


def running_mean(img, vert_slices, wsize):
    """
    Computes the horizontal moving histogram of an image

    :param img: The binary image (ch = 2)
    :param vert_slices: Number of vertical slices
    :param wsize: The window size
    :return: The computed histograms
    """
    size = img.shape[0] / vert_slices
    result = np.zeros(shape=(vert_slices, img.shape[1]), dtype=np.float)

    for i in np.arange(vert_slices):
        start = int(i * size)
        end = int((i + 1) * size)
        vertical_mean = np.mean(img[start:end], axis=0)

        for j in np.arange(wsize / 2):
            vertical_mean = np.insert(vertical_mean, 0, vertical_mean[0])
            vertical_mean = np.insert(vertical_mean, len(vertical_mean), vertical_mean[-1])

        window_sum = np.cumsum(vertical_mean)
        result[i, :] = (window_sum[wsize:] - window_sum[:-wsize]) / wsize

    return result



