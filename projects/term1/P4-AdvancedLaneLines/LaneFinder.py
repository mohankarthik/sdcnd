import numpy as np
import cv2
import calib
import preprocess
from glob import glob
import matplotlib.pyplot as plt


class HistogramSearch:
    def __init__(self, vertical_slices):
        """
        Initializes the Histogram class

        :param vertical_slices: Number of vertical slices by which to split the image by
        """

        # The number of vertical slices in the histogram processing
        self.vertical_slices = vertical_slices

        # The markings for each cycle
        self.markings = []

        # The shape of the image
        self.size = None

    def find(self, img, base_lane_width, lane_width_confident):
        """
        Finds lane fit equations in the 2nd degree from a bird's eye view of the road

        :param img: Bird's eye view of the road, where the lane lines are parallel
        :param base_lane_width: The lane width at the base
        :param lane_width_confident: A boolean to say if the application is confident of it's lane width
        :return: Left and right markings
        """
        # Save the shape
        self.size = img.shape

        # Compute the histogram
        histogram = preprocess.running_mean(img, self.vertical_slices, 50)

        # Search each of the histograms for the lanes
        raw_markings = []
        self.markings = []
        dbg_intervals = []
        dbg_confidences = []

        for i in np.arange(self.vertical_slices - 1, -1, -1):
            # Get the intervals and lane width from previous values
            intervals, lane_width, lane_width_confidence, left_confidence, right_confidence = \
                self.__get_expected_interval_and_width(img.shape[1], lane_width_confident)
            dbg_intervals.append(intervals)
            dbg_confidences.append([lane_width_confidence, left_confidence, right_confidence])

            # If we weren't able to find the lane width, take the given base value
            if lane_width is None:
                lane_width = base_lane_width

            # Persistent values across the loop
            prev_pos = 0
            start = 0
            found = []
            features = np.argwhere(histogram[i] > 0)

            for val in features:
                # Find if the value lies within any of the intervals
                if (intervals[0][0] <= val <= intervals[0][1]) or (intervals[1][0] <= val <= intervals[1][1]):
                    if val > prev_pos + 10:
                        # If we've already started to find a interval
                        if start:
                            # Append the results with this interval and start a new interval
                            found.append(np.mean([start, prev_pos]))
                            start = val
                        else:
                            start = val
                    prev_pos = val

                else:
                    # If the value does not lie in any of the expected intervals, then go to the next value,
                    # But before that, in case we 've a lane in the queue, we should add it
                    if start:
                        found.append(np.mean([start, prev_pos]))
                        start = 0

            # Add the final lane marking (if we've a valid lane in the queue)
            if start:
                found.append(np.mean([start, prev_pos]))

            # # Find two values that are probable lane candidates from the list of found
            candidates = self.__find_prob_markings(found, lane_width,
                                                   (np.mean(intervals[0]), np.mean(intervals[1])),
                                                   lane_width_confidence, left_confidence, right_confidence)

            # Add to the global list
            raw_markings.append(found)
            self.markings.append(candidates)

        return self.markings

    # ------------- STATIC FUNCTIONS ------------- #
    def __get_expected_interval_and_width(self, shape, confident):
        """
        Gets the expected lane width and marking interval from the previous vertical slice

        :param shape: Shape of the image
        :param confident: If this histogram search is from scratch, or if it's part of recovery
        :return: The interval and the width
        """
        intervals = [[0, shape / 2], [shape / 2, shape]]
        lane_width = None
        if confident:
            lane_width_confidence = 0
        else:
            lane_width_confidence = 100
        left_confidence = 100
        right_confidence = 100

        # Search from the most recent
        left_found = right_found = False
        for i in np.arange(len(self.markings) - 1, -1, -1):
            # Check if this one is not None
            if not left_found and self.markings[i][0] is not None:
                intervals[0] = [self.markings[i][0] - 200, self.markings[i][0] + 200]
                left_found = True
            if not right_found and self.markings[i][1] is not None:
                intervals[1] = [self.markings[i][1] - 200, self.markings[i][1] + 200]
                right_found = True
            if left_found and right_found:
                break
            if not left_found:
                left_confidence += 25
            if not right_found:
                right_confidence += 25
            if confident:
                lane_width_confidence += 25
            else:
                lane_width_confidence += 50

        if left_found and right_found:
            lane_width = np.mean(intervals[1]) - np.mean(intervals[0])

        return intervals, lane_width, lane_width_confidence, left_confidence, right_confidence

    def __find_prob_markings(self, curr_finds, lane_width, curr_range,
                             lane_width_confidence, left_confidence, right_confidence):
        """
        Finds probable lane markings from all the marking founds

        :param curr_finds: All the markings found in a single vertical slice histogram
        :param lane_width: The expected lane width at this histogram
        :param curr_range: The currently expected mean of lane marking for each lane
        :param lane_width_confidence: The predicted error in the lane width
        :param left_confidence: The predicted error in the left marking mean
        :param right_confidence: The predicted error in the right marking mean
        :return: Returns the lane co-ordinates if it was found, or returns None
        """
        answer = [None, None]

        if len(curr_finds):
            # If this is not the first histogram (bottom row), test against the previous
            # histogram values to find the correct markings
            answer = self.__check_prev_value(curr_finds, curr_range,
                                             left_confidence, right_confidence)

            # Cross verify the values against the lane width if we've both the sides
            if answer[0] is not None and answer[1] is not None:
                lane_answer = self.__check_width(answer, lane_width, lane_width_confidence)
                if lane_answer != answer:
                    answer = [None, None]

        return answer

    # ------------- STATIC FUNCTIONS ------------- #
    @staticmethod
    def __check_width(curr_finds, lane_width, lane_width_confidence):
        """
        Validates the found markings against the expected lane width

        :param curr_finds: The currently found markings
        :param lane_width: The currently expected lane width
        :param lane_width_confidence: The currently expected error in the lane width
        :return: None, or the valid found marking as a list of 2 values, one each for left and right lane
        """
        answer = [None, None]

        results = np.ones(shape=(len(curr_finds), len(curr_finds)), dtype=np.float)
        results *= np.infty

        for i in np.arange(len(curr_finds)):
            if curr_finds[i] is not None:
                for j in np.arange(i + 1, len(curr_finds)):
                    if curr_finds[j] is not None:
                        results[i][j] = np.abs((curr_finds[j] - curr_finds[i]) - lane_width)

        if results.min() < lane_width_confidence:
            # If the lane width between any of the two curr_finds matches the expected lane width,
            # then hurray, we've the answer
            answer = np.unravel_index(results.argmin(), results.shape)
            answer = [curr_finds[answer[0]], curr_finds[answer[1]]]

        return answer

    @staticmethod
    def __check_prev_value(curr_finds, curr_range, left_confidence, right_confidence):
        """
        Validates the finds against the previously vertical slice

        :param curr_finds: The currently found markings
        :param curr_range: Current expected marking mean for each lane
        :param left_confidence: Current confidence of the left lane search
        :param right_confidence: Current confidence of the right lane search
        :return: The markings or None if they are not valid in an list of 2 values
        """
        answer = [None, None]

        results = np.ones(shape=(len(curr_finds), 2), dtype=np.float)
        results *= np.infty

        for i in np.arange(len(curr_finds)):
            results[i][0] = np.abs(curr_finds[i] - curr_range[0])
            results[i][1] = np.abs(curr_finds[i] - curr_range[1])

        # Find if any of the markings are valid for one of the sides
        min_args = np.argmin(results, 0)
        minimums = np.min(results, 0)
        if minimums[0] < left_confidence:
            answer[0] = curr_finds[min_args[0]]
        if minimums[1] < right_confidence:
            answer[1] = curr_finds[min_args[1]]
        return answer


class LaneFinder:
    def __init__(self, base_lane_width=700, cam_calibration=None, debug_level=0, debug_loc='./'):
        """
        Initializes the LaneFinder class

        :param base_lane_width: The width of the lane in pixels
        :param cam_calibration: Camera calibration parameter
        :param debug_level: Debug level, 0 - 3
        :param debug_loc: Location for dumping diagnostic data
        """

        # Constants
        self.constants = {
            'HIST_VERT_SLICES': 20,  # Number of vertical slices in the histogram detection
            'FIT_ERR_THRESHOLD': 5000,  # Fit error from current to previous frame that is tolerated
            'FIT_AVG_WEIGHT': 0.5,  # Cap for the weighted fit averaging for current fit. As this is increased, more
                                    # weight is given to the current fit.
            'CONFIDENCE_THRESHOLD': 0.05  # Threshold for minimum confidence in each lane's detection
        }

        # The camera calibration parameters
        self.cam_calibration = cam_calibration

        # The original image's shape
        self.img_sz = None

        # Histogram lane finder
        self.hist_finder = HistogramSearch(self.constants['HIST_VERT_SLICES'])

        # Frame memory related variables
        self.frame_mem_max = 5
        self.frame_memory = None
        self.frame_avail = np.zeros(shape=self.frame_mem_max, dtype=np.bool)
        self.frame_idx = 0

        # Typical lane widths at each of the histogram slices, we initialize the base width alone
        self.lane_width = base_lane_width

        # The currently used ROI
        self.ROI = None
        self.roi_stable = False

        # The current fit values
        self.fit_values = [None, None]
        self.fit_confidence = 0
        self.fit_err = 0

        # The current bottom lane start position
        self.find_mode = 'hist'
        self.lane_bottom = [None, None]
        self.lane_confidence = None

        # Diagnostics
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.debug_level = debug_level
        self.debug_loc = debug_loc
        self.cnt = 0
        self.show = None
        self.plot = None

    def process(self, frame):
        """
        Main pipeline function to detect the lane lines given an image

        :param frame: The current frame
        :return: Overlayed frame with the lane lines
        """
        # Save the shape of the original image
        self.img_sz = frame.shape

        # Compute the transformation co-ordinates
        transform_dst = np.array([
            [0, frame.shape[0]],
            [0, 0],
            [frame.shape[1], 0],
            [frame.shape[1], frame.shape[0]]
        ])

        # Find a probable region of interest
        self.__find_roi(frame)

        # Initialize the frame memory with the frame shape with ROI is unstable
        if not self.roi_stable or self.frame_memory is None:
            self.frame_memory = np.zeros(shape=(self.frame_mem_max, frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Apply the distortion correction to the raw image.
        if self.cam_calibration is not None:
            dist_correct = calib.undistort(frame, self.cam_calibration)
        else:
            dist_correct = np.copy(frame)

        # Transform the perspective
        perspective_transformer = preprocess.PerspectiveTransformer(np.float32(self.ROI), np.float32(transform_dst))
        birdseye = perspective_transformer.transform(dist_correct)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        white, yellow, color, sobel, combined, binary = self.__threshold_image(birdseye, self.debug_level > 1)

        # Add the image to the frame memory
        self.frame_memory[self.frame_idx] = binary
        self.frame_avail[self.frame_idx] = True
        # Allow more images to be added only after the ROI is stable.
        if self.roi_stable:
            self.frame_idx = (self.frame_idx + 1) % self.frame_mem_max

        # Compute a motion blur across the frame memory
        motion = np.zeros(shape=(self.img_sz[0], self.img_sz[1]), dtype=np.uint8)
        for idx, img in enumerate(self.frame_memory):
            if self.frame_avail[idx]:
                motion[(img > .5)] = 1

        if self.fit_err > self.constants['FIT_ERR_THRESHOLD'] or not self.roi_stable:
            # Search for the lanes using sliding windows
            lane_markings = self.hist_finder.find(motion, self.lane_width, self.roi_stable)
            self.find_mode = 'hist'
        else:
            lane_markings = self.__get_markings_from_fit()
            self.find_mode = 'prev'

        # Get a fit from the lane markings
        confidences, fits, masks = self.__get_fit(motion, lane_markings)

        # Smoothen the fit
        for i in np.arange(2):
            # If we don't have a value from previous frame, just use the current one and hope for the best
            # If the ROI is not stable, then also just use the current one and don't smoothen
            if self.fit_values[i] is None or not self.roi_stable:
                self.fit_err = 0
                self.fit_values[i] = fits[i]
            else:
                # Check if we've got a valid value now
                if fits[i] is not None:
                    # Calculate the error from the previous frame
                    self.fit_err = np.mean((self.fit_values[i] - fits[i]) ** 2)

                    # If error is within expected values, then average the fit
                    if self.fit_err < self.constants['FIT_ERR_THRESHOLD'] and confidences[i] > 0.05:
                        # Average out based on the confidence
                        prev_confidence = self.constants['FIT_AVG_WEIGHT'] + \
                                          ((1 - self.constants['FIT_AVG_WEIGHT']) * self.fit_confidence /
                                           (self.fit_confidence + confidences[i]))
                        curr_confidence = (1 - self.constants['FIT_AVG_WEIGHT']) * \
                                          (confidences[i] / (self.fit_confidence + confidences[i]))
                        self.fit_values[i] = (self.fit_values[i] * prev_confidence) + \
                                              (fits[i] * curr_confidence)
                    elif self.find_mode =='hist' and \
                                    self.fit_err < (2 * self.constants['FIT_ERR_THRESHOLD']) and \
                                    confidences[i] > self.constants['CONFIDENCE_THRESHOLD']:
                        self.fit_err = 0
                        self.fit_values[i] = fits[i]

        # Update the confidences
        if fits[0] is not None and fits[1] is not None:
            prev_confidence = self.constants['FIT_AVG_WEIGHT'] + \
                              ((1 - self.constants['FIT_AVG_WEIGHT']) * self.fit_confidence /
                               (self.fit_confidence + np.mean(confidences)))
            curr_confidence = (1 - self.constants['FIT_AVG_WEIGHT']) * np.mean(confidences) / \
                              (self.fit_confidence + np.mean(confidences))
            self.fit_confidence = (self.fit_confidence * prev_confidence) + \
                                  (np.mean(confidences) * curr_confidence)

        # Draw the lanes
        if self.fit_values[0] is not None and self.fit_values[1] is not None:
            overlay = self.__draw_overlay(binary)
        else:
            overlay = np.zeros_like(frame)

        # Transform back to the original perspective and add the overlay onto the original image
        overlay_perspective = perspective_transformer.inverse_transform(overlay)
        result = cv2.addWeighted(overlay_perspective, .7, frame, 1., 0.)

        # Compute the curvature
        curvature = self.__compute_curvature(np.mean(self.fit_values, 0), self.img_sz[0] / 2)
        str_curv = 'Curvature = ' + str(np.round(curvature, 2))
        cv2.putText(result, str_curv, (500, 400), self.font, 1, (255, 0, 0), 1)

        # Compute the offset from the middle
        dist_offset = (((self.lane_bottom[1] - self.lane_bottom[0]) - (self.img_sz[1] / 2)) /
                       (self.lane_bottom[1] - self.lane_bottom[0]))
        dist_offset = np.round(dist_offset * 100, 2)
        str_offset = 'Lane deviation: ' + str(dist_offset) + '%.'
        if dist_offset > 20.:
            cv2.putText(result, str_offset, (500, 700), self.font, 1, (255, 0, 0), 1)
        else:
            cv2.putText(result, str_offset, (500, 700), self.font, 1, (0, 255, 255), 1)

        """
        DEBUG CODE
        """
        if self.debug_level > 0:
            # Write up the confidence
            str_conf = 'Confidence = ' + str(np.round(self.fit_confidence, 2))
            cv2.putText(result, str_conf, (30, 120), self.font, 1, (255, 0, 0), 2)

            # Raw confidence
            str_conf = 'Raw confidence: Left = ' + str(np.round(confidences[0], 2)) + '; Right = ' + str(np.round(confidences[1], 2))
            cv2.putText(result, str_conf, (30, 150), self.font, 1, (255, 0, 0), 2)

            # Error in fit values
            str_err = 'Error in fit values = ' + str(np.round(self.fit_err, 2))
            cv2.putText(result, str_err, (30, 180), self.font, 1, (255, 0, 0), 2)

            # Draw boxes for each found marking
            box_marked = np.copy(birdseye)
            size = frame.shape[0] / self.constants['HIST_VERT_SLICES']
            for idx, marking in enumerate(lane_markings):
                start = int(size * (self.constants['HIST_VERT_SLICES'] - idx - 1))
                end = int(size * (self.constants['HIST_VERT_SLICES'] - idx))
                if marking[0] is not None:
                    cv2.rectangle(box_marked, (int(marking[0]) - 100, end), (int(marking[0]) + 100, start), (255), 10)
                if marking[1] is not None:
                    cv2.rectangle(box_marked, (int(marking[1]) - 100, end), (int(marking[1]) + 100, start), (255), 10)

            # Mark the ROI in the image
            marked = np.copy(frame)
            cv2.line(marked,
                     (self.ROI[0][0], self.ROI[0][1]),
                     (self.ROI[1][0], self.ROI[1][1]),
                     (255, 0, 0), 10)
            cv2.line(marked,
                     (self.ROI[2][0], self.ROI[2][1]),
                     (self.ROI[3][0], self.ROI[3][1]),
                     (255, 0, 0), 10)
            cv2.line(marked,
                     (self.ROI[0][0], self.ROI[0][1]),
                     (self.ROI[3][0], self.ROI[3][1]),
                     (255, 0, 0), 10)
            cv2.line(marked,
                     (self.ROI[1][0], self.ROI[1][1]),
                     (self.ROI[2][0], self.ROI[2][1]),
                     (255, 0, 0), 10)
            str_conf = 'ROI Stable = ' + str(self.roi_stable)
            cv2.putText(marked, str_conf, (30, 60), self.font, 1, (255, 0, 0), 2)
            str_conf = 'Lane Width = ' + str(self.lane_width)
            cv2.putText(marked, str_conf, (30, 90), self.font, 1, (255, 0, 0), 2)

            diagScreen = np.zeros((1080, 1920, 3), dtype=np.uint8)

            diagScreen[0:720, 0:1280] = cv2.resize(result, (1280, 720), interpolation=cv2.INTER_AREA)

            marked = cv2.copyMakeBorder(marked, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            marked = cv2.resize(marked, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[0:240, 1280:1600] = marked

            birdseye = cv2.copyMakeBorder(birdseye, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            birdseye = cv2.resize(birdseye, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[240:480, 1280:1600] = birdseye

            box_marked = cv2.copyMakeBorder(box_marked, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            box_marked = cv2.resize(box_marked, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[480:720, 1280:1600] = box_marked

            motion = cv2.cvtColor(motion, cv2.COLOR_GRAY2RGB) * 255.
            motion = cv2.copyMakeBorder(motion, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            motion = cv2.resize(motion, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[720:960, 0:320] = motion

            mask = cv2.cvtColor(masks[0] + masks[1], cv2.COLOR_GRAY2RGB) * 255.
            mask = cv2.copyMakeBorder(mask, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            mask = cv2.resize(mask, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[720:960, 320:640] = mask

            overlay = cv2.copyMakeBorder(overlay, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            overlay = cv2.resize(overlay, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[720:960, 960:1280] = overlay

            overlay_perspective = cv2.copyMakeBorder(overlay_perspective, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 0))
            overlay_perspective = cv2.resize(overlay_perspective, (320, 240), interpolation=cv2.INTER_AREA)
            diagScreen[720:960, 1280:1600] = overlay_perspective

            # Show off the plots if required
            if self.debug_level > 1:
                fig = plt.figure()
                fig.add_subplot(3, 4, 1), plt.imshow(frame), plt.title('Original')
                fig.add_subplot(3, 4, 2), plt.imshow(dist_correct), plt.title('Distortion corrected')
                fig.add_subplot(3, 4, 3), plt.imshow(marked), plt.title('ROI marked')
                fig.add_subplot(3, 4, 4), plt.imshow(birdseye), plt.title('Bird\'s view')
                fig.add_subplot(3, 4, 5), plt.imshow(binary, cmap='gray'), plt.title('Thresholded')
                fig.add_subplot(3, 4, 6), plt.imshow(box_marked, cmap='gray'), plt.title('Histogram find')
                fig.add_subplot(3, 4, 7), plt.imshow(masks[0] + masks[1], cmap='gray'), plt.title('Masked pixels')
                fig.add_subplot(3, 4, 8), plt.imshow(overlay), plt.title('Overlay')
                fig.add_subplot(3, 4, 9), plt.imshow(overlay_perspective), plt.title('Perspective overlay')
                fig.add_subplot(3, 4, 10), plt.imshow(result), plt.title('Result')
                plt.show()

            if self.debug_level > 2:
                # Construct a diagnostic view
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-00orig.jpg',
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-01roi.jpg',
                            cv2.cvtColor(marked, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-02distortion.jpg',
                            cv2.cvtColor(dist_correct, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-03birdseye.jpg',
                            cv2.cvtColor(birdseye, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-04threshold.jpg',
                            cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR) * 255.)
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-05motion.jpg',
                            cv2.cvtColor(motion, cv2.COLOR_GRAY2BGR) * 255.)
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-06select.jpg',
                            cv2.cvtColor(box_marked, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-07pixels.jpg',
                            cv2.cvtColor(masks[0] + masks[1], cv2.COLOR_GRAY2BGR) * 255.)
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-08overlay.jpg',
                            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-09overlay_p.jpg',
                            cv2.cvtColor(overlay_perspective, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-10result.jpg',
                            cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                cv2.imwrite(self.debug_loc + '/{0:04d}'.format(self.cnt) + '-11diag.png',
                            cv2.cvtColor(diagScreen, cv2.COLOR_RGB2BGR))
                self.cnt += 1

        return result

    # ------------- PRIVATE FUNCTIONS ------------- #
    def __get_markings_from_fit(self):
        """
        Gets markings from a fit

        :return: The mean markings where the lanes might be found
        """
        # Generate the x and y for each lane boundary
        right_y = np.arange(self.constants['HIST_VERT_SLICES'] - 1, -1, -1) * self.img_sz[0] / self.constants['HIST_VERT_SLICES']
        right_x = self.fit_values[1][0] * right_y ** 2 + self.fit_values[1][1] * right_y + self.fit_values[1][2]

        left_y = np.arange(self.constants['HIST_VERT_SLICES'] - 1, -1, -1) * self.img_sz[0] / self.constants['HIST_VERT_SLICES']
        left_x = self.fit_values[0][0] * left_y ** 2 + self.fit_values[0][1] * left_y + self.fit_values[0][2]

        markings = [[left_x[i], right_x[i]] for i in np.arange(self.constants['HIST_VERT_SLICES'])]

        return markings

    def __draw_overlay(self, birdseye):
        """
        Draws the lanes specified by the fits on top of the given image

        :param birdseye: The birdseye image
        :return: Overlay image
        """
        # Generate the x and y for each lane boundary
        right_y = np.arange(11) * self.img_sz[0] / 10
        right_x = self.fit_values[1][0] * right_y ** 2 + self.fit_values[1][1] * right_y + self.fit_values[1][2]

        left_y = np.arange(11) * self.img_sz[0] / 10
        left_x = self.fit_values[0][0] * left_y ** 2 + self.fit_values[0][1] * left_y + self.fit_values[0][2]

        # Save the bottom values
        if self.lane_bottom[0] is not None:
            self.lane_bottom[0] = (self.lane_bottom[0] * .9) + (left_x[-1] * .1)
        else:
            self.lane_bottom[0] = left_x[-1]
        if self.lane_bottom[1] is not None:
            self.lane_bottom[1] = (self.lane_bottom[1] * .9) + (right_x[-1] * .1)
        else:
            self.lane_bottom[1] = right_x[-1]

        # Cast the points into a form that's easy for cv2.fillPoly
        temp = np.zeros_like(birdseye).astype(np.uint8)
        overlay = np.dstack((temp, temp, temp))
        left_pts = np.array([np.transpose(np.vstack([left_x, left_y]))])
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, right_y])))])
        pts = np.hstack((left_pts, right_pts))

        # Plot the overlay
        cv2.fillPoly(overlay, np.int_([pts]), (0, 128, 0))

        # Draw the lane onto the warped blank image
        self.__draw_line(overlay, np.int_(left_pts), (0, 255, 0))
        self.__draw_line(overlay, np.int_(right_pts), (0, 255, 0))

        return overlay

    def __get_fit(self, img, markings):
        """
        Gets the polynomial fit given a set of markings

        :param img: The image on which to find the fit on using the markings. This should be a binary thresholded image
        :param markings: The markings (mean value of possible lanes)
        :return: [Left_fit, Right_fit]
        """
        left_image = np.zeros_like(img)
        right_image = np.zeros_like(img)
        size = img.shape[0] / self.constants['HIST_VERT_SLICES']

        # Create masks
        for idx, marking in enumerate(markings):
            start = int(size * (self.constants['HIST_VERT_SLICES'] - idx - 1))
            end = int(size * (self.constants['HIST_VERT_SLICES'] - idx))
            if marking[0] is not None:
                pts = np.array([[[int(marking[0]) - 100, end],
                                [int(marking[0]) - 100, start],
                                [int(marking[0]) + 100, start],
                                [int(marking[0]) + 100, end]]], dtype=np.int32)
                cv2.fillPoly(left_image, pts, 255)
            if marking[1] is not None:
                pts = np.array([[[int(marking[1]) - 100, end],
                             [int(marking[1]) - 100, start],
                             [int(marking[1]) + 100, start],
                             [int(marking[1]) + 100, end]]], dtype=np.int32)
                cv2.fillPoly(right_image, pts, 255)

        # apply the masks
        left_image = cv2.bitwise_and(left_image, img)
        right_image = cv2.bitwise_and(right_image, img)

        # Recompute the bottom slice
        start = int(size * (self.constants['HIST_VERT_SLICES'] - 1))
        end = int(size * (self.constants['HIST_VERT_SLICES']))

        if len(np.argwhere(left_image > .5)) < 5000 and self.lane_bottom[0] is not None:
            left_image[start:end, self.lane_bottom[0] - 10:self.lane_bottom[0] + 10] = 1
        if len(np.argwhere(right_image > .5)) < 5000 and self.lane_bottom[1] is not None:
            right_image[start:end, self.lane_bottom[1] - 10:self.lane_bottom[1] + 10] = 1

        # Extract the left and right points
        left_fit = right_fit = None
        left_values = np.argwhere(left_image > .5)
        if len(left_values) >= 10:
            all_x = left_values.T[0]
            all_y = left_values.T[1]
            left_fit = np.polyfit(all_x, all_y, 2)

        right_values = np.argwhere(right_image > .5)
        if len(right_values) >= 10:
            all_x = right_values.T[0]
            all_y = right_values.T[1]
            right_fit = np.polyfit(all_x, all_y, 2)

        # Calculate confidence as compared to the maximum number of image pixels we could have got
        confidences = [len(left_values) / (100 * img.shape[0]), len(right_values) / (100 * img.shape[0])]

        return confidences, [left_fit, right_fit], [left_image, right_image]

    def __find_roi(self, img):
        """
        Finds the ROI either from scratch, or from the previous fit

        :param img: Image from which ROI should be found
        :return: The ROI vertices
        """
        # If there is an existing fit, then use that to find the ROI
        if self.fit_values[0] is not None and self.fit_values[1] is not None:
            # Get the fit values at start and end to compute the ROI for the next frame
            left_bot_x = self.__evaluate_curve(img.shape[0], self.fit_values[0])
            left_top_x = self.__evaluate_curve(0, self.fit_values[0])
            right_bot_x = self.__evaluate_curve(img.shape[0], self.fit_values[1])
            right_top_x = self.__evaluate_curve(0, self.fit_values[1])

            # Calculate the correction
            corr_lef = left_top_x - left_bot_x
            corr_right = right_top_x - right_bot_x

            # Check if the ROI needs a translation or a shape change
            if (self.img_sz[1] - right_top_x) < (self.img_sz[1] * 0.1):
                corr_lef -= self.img_sz[1] * 0.05
                corr_right -= self.img_sz[1] * 0.05
            if left_top_x > (self.img_sz[1] * 0.1):
                corr_lef += self.img_sz[1] * 0.05
                corr_right += self.img_sz[1] * 0.05

            # Update the ROI
            if self.roi_stable:
                self.roi_stable = True
                if (self.img_sz[1] * 0.15) < np.abs(corr_lef) < (self.img_sz[1] * 0.2):
                    self.ROI[1][0] += (left_top_x - left_bot_x) * 0.05
                    self.roi_stable = False
                if (self.img_sz[1] * 0.15) < np.abs(corr_lef) < (self.img_sz[1] * 0.2):
                    self.ROI[2][0] += (right_top_x - right_bot_x) * 0.05
                    self.roi_stable = False
            else:
                self.roi_stable = True
                if np.abs(corr_lef) > (self.img_sz[1] * 0.1):
                    self.ROI[1][0] += (left_top_x - left_bot_x) * 0.15
                    self.roi_stable = False
                if np.abs(corr_right) > (self.img_sz[1] * 0.1):
                    self.ROI[2][0] += (right_top_x - right_bot_x) * 0.15
                    self.roi_stable = False

            # Update the lane width
            self.lane_width = right_bot_x - left_bot_x
        else:
            # Try to get the ROI by finding lanes using the hough method
            try:
                bnw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                flt = preprocess.guassian_blur(bnw, 21)
                cny = preprocess.canny(flt, 40, 50)
                ROI = np.array([[self.img_sz[1] * 0.0, self.img_sz[0] * 1.0],
                                [self.img_sz[1] * 0.4, self.img_sz[0] * 0.65],
                                [self.img_sz[1] * 0.6, self.img_sz[0] * 0.65],
                                [self.img_sz[1] * 1.0, self.img_sz[0] * 1.0]])
                roi = preprocess.region_of_interest(cny, np.int32([ROI]))
                verts = preprocess.hough_lines(roi, 1, np.pi/48, 50, 1, 60, ROI[1][1])

                # If we've not found the lanes
                if verts[0][1] == 0 or verts[1][1] == 0 or verts[2][1] == 0 or verts[3][1] == 0:
                    raise ValueError

                # If we've found the lanes, spread around in the x axis and make the ROI have some room
                verts[0][0] -= 200
                verts[1][0] -= 200
                verts[2][0] += 200
                verts[3][0] += 200

                # Update the global
                self.ROI = np.copy(verts)

            except:
                # When an error occurs, let's just return a black ROI with 0 confidence
                self.ROI = np.array([[self.img_sz[1] * 0.0, self.img_sz[0] * 1.0],
                                     [self.img_sz[1] * 0.4, self.img_sz[0] * 0.65],
                                     [self.img_sz[1] * 0.6, self.img_sz[0] * 0.65],
                                     [self.img_sz[1] * 1.0, self.img_sz[0] * 1.0]])
                self.ROI = self.ROI.astype(np.int)

    # ------------- STATIC FUNCTIONS ------------- #
    @staticmethod
    def __compute_curvature(pol_a, y_pt):
        """
        Computes the curvature given a line fit

        :param pol_a: The fit values (quadratic)
        :param y_pt: The y location at which the radius of curvature must be measured
        :return The curvature of the curve pointed by pol_a at y_pt
        """
        A = pol_a[0]
        B = pol_a[1]
        result = (1 + (2 * A * y_pt + B) ** 2) ** 1.5 / 2 / A
        return result

    @staticmethod
    def __evaluate_curve(y, fit):
        """
        Evaluates a curve fit at some y

        :param y: The y at which the curve fit should be evaluated
        :param fit: The curve parameters
        :return: The value of x at the position y
        """
        return fit[0] * (y ** 2) + fit[1] * y + fit[2]

    @staticmethod
    def __draw_line(img, pts, color):
        """
        Draws a line given a series of points

        :param img: The image on which line should be drawn
        :param pts: Points to draw the image
        :param color: Color of the line
        :return: None
        """
        pts = np.int_(pts)
        for i in np.arange(len(pts[0]) - 1):
            x1 = pts[0][i][0]
            y1 = pts[0][i][1]
            x2 = pts[0][i + 1][0]
            y2 = pts[0][i + 1][1]
            cv2.line(img, (x1, y1), (x2, y2), color, 50)

    @staticmethod
    def __threshold_image(img, draw=False):
        """
        Thresholds an image based on various criteria

        :param img: Image to be thresholded
        :param draw: If the results should be plotted
        :return: Returns a thresholded image
        """
        # Compute color thresholds
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        white = cv2.inRange(hsv, (20, 0, 180), (255, 80, 255))
        white = preprocess.median_blur(white, 11)
        yellow = cv2.inRange(hsv, (0, 80, 100), (50, 255, 255))
        color = cv2.bitwise_or(white, yellow)

        # Compute Sobel thresholds on the L channel
        sobel_x = preprocess.Sobel.absolute_thresh(hsv[:, :, 1], orientation='x', sobel_kernel=5, threshold=(50, 255))
        sobel_y = preprocess.Sobel.absolute_thresh(hsv[:, :, 1], orientation='y', sobel_kernel=5, threshold=(200, 255))
        sobel_l = np.copy(cv2.bitwise_or(sobel_x, sobel_y))

        # Compute Sobel thresholds on the S channel
        sobel_x = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='x', sobel_kernel=5, threshold=(50, 255))
        sobel_y = preprocess.Sobel.absolute_thresh(hsv[:, :, 2], orientation='y', sobel_kernel=5, threshold=(200, 255))
        sobel_s = np.copy(cv2.bitwise_or(sobel_x, sobel_y))

        # Combine the Sobel and filter it
        sobel = preprocess.median_blur(cv2.bitwise_or(sobel_l, sobel_s), 11)

        # Combine all the thresholds and form a binary image
        combined = np.zeros_like(sobel)
        combined[(color >= .5) | (sobel >= .5)] = 1
        result = preprocess.median_blur(combined, 5)

        # Clear out the last few pixels to remove the car bonnet
        result[result.shape[0]*0.90:][:] = 0

        if draw:
            fig = plt.figure()
            fig.add_subplot(331), plt.imshow(img)
            fig.add_subplot(332), plt.imshow(white, cmap='gray'), plt.title('White')
            fig.add_subplot(333), plt.imshow(yellow, cmap='gray'), plt.title('Yellow')
            fig.add_subplot(334), plt.imshow(color, cmap='gray'), plt.title('Color')
            fig.add_subplot(335), plt.imshow(sobel_l, cmap='gray'), plt.title('Sobel L')
            fig.add_subplot(336), plt.imshow(sobel_s, cmap='gray'), plt.title('Sobel S')
            fig.add_subplot(337), plt.imshow(sobel, cmap='gray'), plt.title('Sobel')
            fig.add_subplot(338), plt.imshow(combined, cmap='gray'), plt.title('Combined')
            fig.add_subplot(339), plt.imshow(result, cmap='gray'), plt.title('Final')
            plt.show()

        return white, yellow, color, sobel, combined, result


if __name__ == "__main__":
    # Calibrate the camera
    calibration = calib.calibrate_camera('camera_cal', (9, 6), (720, 1280, 3))

    test_mode = 'video'

    if test_mode == 'image':
        # Find the lanes
        from scipy.misc import imread, imsave
        images = glob('test_images/*')
        for idx, img_path in enumerate(images):
            img = imread(img_path)
            ld = LaneFinder(cam_calibration=calibration, debug_loc='output_images', debug_level=2)
            res = ld.process(img)
            imsave('output_images/test' + str(idx) + '.png', res)
    else:
        from moviepy.editor import VideoFileClip
        ld = LaneFinder(cam_calibration=calibration)
        project_output = 'project_video_out.mp4'
        clip1 = VideoFileClip('project_video.mp4')
        project_clip = clip1.fl_image(ld.process)
        project_clip.write_videofile(project_output, audio=False)