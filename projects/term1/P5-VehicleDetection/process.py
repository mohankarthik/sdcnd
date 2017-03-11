import matplotlib.pyplot as plt
import pickle
import time
import glob
from moviepy.editor import VideoFileClip
from utils import *


class Detector:
    def __init__(self, classifier, scaler, parameters, debug_level=0):
        # Class parameters
        self.windows = None
        self.classifier = classifier
        self.scaler = scaler
        self.parameters = parameters

        # Size of the image
        self.img_size = None

        # Motion blur
        self.motion = None
        self.motion_idx = 0
        self.curr_motion_imgs = 0
        self.max_motion = 5
        self.old_boxes = []

        # Debug variables
        self.debug_level = debug_level
        self.debug_cnt = 0

    def process(self, image):
        """
        Processes an image
        """
        # Save the image size
        if self.img_size is None:
            self.img_size = image.shape

        # Form up the windows if it's not already
        if self.windows is None:
            self.windows = get_windows(image, x_range=(None, None), y_range=(400, 500),
                                       window_size=(96, 96), overlap=(0.75, 0.75))
            self.windows += get_windows(image, x_range=(None, None), y_range=(400, 500),
                                        window_size=(144, 144), overlap=(0.75, 0.75))
            self.windows += get_windows(image, x_range=(None, None), y_range=(430, 550),
                                        window_size=(192, 192), overlap=(0.75, 0.75))
            self.windows += get_windows(image, x_range=(None, None), y_range=(460, 580),
                                        window_size=(192, 192), overlap=(0.75, 0.75))

        # Normalize image
        norm = image.astype(np.float32) / 255

        # Get possible windows
        t1 = 0
        if self.debug_level > 0:
            t1 = time.clock()
        hot_windows = find_cars(norm, self.windows, self.classifier, self.scaler, self.parameters)
        if self.debug_level > 0:
            print('Found cars in {} seconds'.format(time.clock() - t1))

        # Average across multiple frames
        avg_windows = self.blur_frames(hot_windows)

        # Find an average width and length for the windows
        results, self.old_boxes = self.blur_boxes(avg_windows, self.old_boxes, image.shape)

        # Combine the windows to form a single window for each car
        avg_window_img = np.copy(image)
        avg_window_img = draw_boxes(avg_window_img, results)

        if self.debug_level >= 1:
            # Draw the raw windows
            raw_window_img = np.copy(image)
            raw_window_img = draw_boxes(raw_window_img, hot_windows)

            # Get the heatmap
            raw_heatmap = create_heatmap(hot_windows, image.shape)
            avg_heatmap = create_heatmap(avg_windows, image.shape)

            # Show it
            fig = plt.figure()
            fig.add_subplot(221), plt.imshow(raw_window_img), plt.title('Raw Windows')
            fig.add_subplot(222), plt.imshow(raw_heatmap), plt.title('Heatmap')
            fig.add_subplot(223), plt.imshow(avg_window_img), plt.title('Motion averaged Windows')
            fig.add_subplot(224), plt.imshow(avg_heatmap), plt.title('Motion averaged Heatmap')
            plt.show()

        return avg_window_img

    # ----- Merging multiple frames for better detection ----- #
    def blur_frames(self, hot_windows):
        """
        Averages the detections across frames creating a motion blur if you will
        """
        # Find the heatmap for this image
        heatmap = create_heatmap(hot_windows, self.img_size)

        if self.motion is None:
            self.motion = np.zeros((self.max_motion, heatmap.shape[0], heatmap.shape[1]), dtype=np.float64)

        self.motion[self.motion_idx] = heatmap
        self.motion_idx += 1
        if self.motion_idx >= self.max_motion:
            self.motion_idx = 0
        self.curr_motion_imgs += 1
        if self.curr_motion_imgs > self.max_motion:
            self.curr_motion_imgs = self.max_motion

        return find_windows_from_heatmap(np.average(self.motion, axis=0),
                                         threshold=float(self.curr_motion_imgs / self.max_motion))

    # ----- Merging boxes across frames for better detection ---- #
    def blur_boxes(self, hot_windows, old_boxes,
                   image_shape):
        hot_boxes = self.__initialize_centers(hot_windows)
        new_boxes = self.__correlate_boxes(hot_boxes, old_boxes)
        filtered_boxes = []

        # Find the new boxes that have a decent confidence (atleast one previous match)
        for new_box in new_boxes:
            if new_box[-1] > 2:
                filtered_boxes.append(new_box)

        # Converts the height width back into (x1, y1), (x2, y2)
        new_windows = []
        for filtered_box in filtered_boxes:
            new_center, new_width, new_height, new_move, new_prob = filtered_box
            new_windows.append(((int(new_center[0] - new_width), int(new_center[1] - new_height)),
                                (int(new_center[0] + new_width), int(new_center[1] + new_height))))

        # Create a heatmap
        heatmap = create_heatmap(new_windows, image_shape)

        # Check if there is any overlap of windows
        if np.unique(heatmap)[-1] >= 2:
            labels = ndi.label(heatmap)[0]
            threshold_heatmap = np.zeros_like(heatmap)
            threshold_heatmap[heatmap >= 2] = 1  # Threshold the heatmap
            threshold_labels = ndi.label(threshold_heatmap)  # Generate labels from the thresholded heatmap

            # Loop through all the labels
            for car_number in range(1, threshold_labels[1] + 1):
                nonzero = (threshold_labels[0] == car_number).nonzero()
                num = labels[nonzero[0][0], nonzero[1][0]]
                labels[labels == num] = 0
            heatmap = labels + threshold_heatmap
            new_windows = find_windows_from_heatmap(heatmap)

        return new_windows, new_boxes

    # ----- PRIVATE FUNCTIONS ----- #
    def __calc_distance(self, a, b):
        """
        Calculates the distance between two points
        """
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def __find_center(self, box):
        """
        Finds the center of a box
        """
        x_start, y_start = box[0]
        x_end, y_end = box[1]
        return (x_start + x_end) / 2.0, (y_start + y_end) / 2.0

    def __find_radius(self, box):
        """
        Find's the radius of the box
        """
        x_start, y_start = box[0]
        x_end, y_end = box[1]
        return (x_end - x_start) / 2, (y_end - y_start) / 2

    def __initialize_centers(self, boxes):
        """
        Initializes an array for the center / radius of the boxes
        :param boxes: A set of boxes (given by (x1, y1), (x2, y2)
        :return: A list of [(center, width, height, motion)]
        """
        result = []
        for box in boxes:
            center = self.__find_center(box)
            width, height = self.__find_radius(box)
            motion = (0, 0)  # motion of the center
            result.append((center, width, height, motion, 1))
        return result

    def __is_near(self, old_center, new_center, old_width, new_width,
                  old_height, new_height):
        """
        Function to check if the old and new boxes are approximately close by

        :return: True / False
        """
        if self.__calc_distance(old_center, new_center) < 5000 and abs(old_width - new_width) < 50 and \
                abs(old_height - new_height) < 50:
            return True
        else:
            return False

    def __average_centers(self, new_center, old_center):
        """
        Simple weighted average of the centers
        """
        w = 2.  # weight parameter
        return ((new_center[0] + old_center[0] * w) / (w + 1),
                (new_center[1] + old_center[1] * w) / (w + 1))

    def __calculate_motion(self, new_center, old_center, old_motion):
        """
        Calculates how much the center has moved based on a weighted equation
        """
        w = 6.  # weight parameter
        return ((new_center[0] - old_center[0] + w * old_motion[0]) / (w + 1),
                (new_center[1] - old_center[1] + w * old_motion[1]) / (w + 1))

    def add_motion_to_center(self, center, motion):
        """
        Adds the movement value to the center to correct the center of jitter
        """
        return center[0] + motion[0] / 5, center[1] + motion[1] / 5

    def __correlate_boxes(self, new_boxes, old_boxes):
        """
        Function to search through the old and new boxes to find matches and then create
        a combined box list with confidences depending on how many matches we found
        """
        fresh_boxes = []
        max_confidence = 40
        temp_new_boxes = copy.copy(new_boxes)
        w = 3  # weight parameter

        # Loop over all old boxes and new boxes and find correlations (similar centers)
        for old_box in old_boxes:
            old_center, old_width, old_height, old_move, old_prob = old_box
            new_boxes = copy.copy(temp_new_boxes)
            if old_prob > 10:
                add_prob = 2
            else:
                add_prob = 1
            found = False
            fresh_box = None
            for new_box in new_boxes:
                new_center, new_width, new_height, new_move, new_prob = new_box
                if self.__is_near(old_center, new_center, old_width, new_width, old_height, new_height):
                    fresh_box = [self.__average_centers(new_center, old_center),
                                 (new_width + w * old_width) / (w + 1),
                                 (new_height + w * old_height) / (w + 1),
                                 self.__calculate_motion(new_center, old_center, old_move),
                                 min(max_confidence, old_prob + add_prob)]
                    # remove the new box from an array
                    temp_new_boxes.remove(new_box)
                    found = True
                    break
            # if no new_box is found, subtract the confidence by 1
            if not found:
                fresh_box = [self.add_motion_to_center(old_center, old_move), old_width, old_height, old_move, old_prob - 1]
            # add the fresh box
            fresh_boxes.append(fresh_box)
        # append the leftover new boxes to old boxes
        fresh_boxes += temp_new_boxes
        # delete if prob = 0
        temp_fresh_boxes = copy.copy(fresh_boxes)
        for box in fresh_boxes:
            if box[-1] <= 0:
                temp_fresh_boxes.remove(box)
        # return the updated old_boxes
        return temp_fresh_boxes


if __name__ == "__main__":
    # Reload the classifier
    pickle_file = 'svc.pickle'
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)
        main_svc = pickle_data['svc']
        main_scaler = pickle_data['scaler']
        main_parameters = pickle_data['parameters']
        del pickle_data  # Free up memory
    print('Model and parameters loaded.')

    mode = 'video'
    if mode == 'images':
        # Read up the test images and process them
        imgs = glob.glob('test_images/test*.jpg')
        for idx, img in enumerate(imgs):
            img = mpimg.imread(img)

            detector = Detector(main_svc, main_scaler, main_parameters, 2)
            img = detector.process(img)

            mpimg.imsave('output_images/test{}.jpg'.format(idx), img)
    else:
        # Draw boxes on a video stream
        white_output = 'test_video_out.mp4'  # New video
        clip1 = VideoFileClip('test_video.mp4')  # Original video
        detector = Detector(main_svc, main_scaler, main_parameters, 0)
        white_clip = clip1.fl_image(detector.process)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
