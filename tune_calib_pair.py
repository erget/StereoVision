#!/bin/env python
'''
A class for working with calibrated stereo camera pairs.
'''

import argparse
from calibrate_stereo import find_files
import calibrate_stereo
import webcams

import cv2

def report_variable(variable_name, variable):
    """Report how often each parameter value was chosen."""
    unique_values = list(set(variable))
    value_frequency = {}
    for value in unique_values:
        value_frequency[variable.count(value)] = value
    frequencies = value_frequency.keys()
    frequencies.sort(reverse=True)
    header = "{} value | Selection frequency".format(variable_name)
    left_column_width = len(header[:-21])
    right_column_width = 21
    print(header)
    print("{}|{}".format("-" * left_column_width, "-" * right_column_width))
    for frequency in frequencies:
        left_column = str(value_frequency[frequency]).center(left_column_width)
        right_column = str(frequency).center(right_column_width)
        print("{}|{}".format(left_column, right_column))

class BadBlockMatcherArgument(Exception):
    """Bad argument supplied for ``block_matcher`` in a ``CalibratedPair``."""
class InvalidBMPreset(BadBlockMatcherArgument):
    """Invalid BM preset."""
class InvalidSearchRange(BadBlockMatcherArgument):
    """Invalid search range."""
class InvalidWindowSize(BadBlockMatcherArgument):
    """Invalid search range."""

class CalibratedPair(webcams.StereoPair):
    """
    A stereo pair of calibrated cameras.

    Should be initialized with a context manager to ensure that the camera
    connections are closed properly.
    """
    def __init__(self, devices,
                 calibration,
                 stereo_bm_preset=cv2.STEREO_BM_BASIC_PRESET,
                 search_range=0,
                 window_size=5):
        """
        Initialize cameras.

        ``devices`` is an iterable of the device numbers. If you want to use the
        ``CalibratedPair`` in offline mode, pass None.
        ``calibration`` is a StereoCalibration object. ``stereo_bm_preset``,
        ``search_range`` and ``window_size`` are parameters for the
        ``block_matcher``.
        """
        if devices:
            super(CalibratedPair, self).__init__(devices)
        #: ``StereoCalibration`` object holding the camera pair's calibration.
        self.calibration = calibration
        self._bm_preset = cv2.STEREO_BM_BASIC_PRESET
        self._search_range = 0
        self._window_size = 5
        #: OpenCV camera type for ``block_matcher``
        self.stereo_bm_preset = stereo_bm_preset
        #: Number of disparities for ``block_matcher``
        self.search_range = search_range
        #: Search window size for ``block_matcher``
        self.window_size = window_size
        #: ``cv2.StereoBM`` object for block matching.
        self.block_matcher = cv2.StereoBM(self.stereo_bm_preset,
                                          self.search_range,
                                          self.window_size)
    def get_frames(self):
        """Rectify and return current frames from cameras."""
        frames = super(CalibratedPair, self).get_frames()
        return self.calibration.rectify(frames)
    def compute_disparity(self, pair):
        """
        Compute disparity from image pair (left, right).

        First, convert images to grayscale if needed. Then pass to the
        ``CalibratedPair``'s ``block_matcher`` for stereo matching.

        If you wish to visualize the image, remember to normalize it to 0-255.
        """
        gray = []
        if pair[0].ndim == 3:
            for side in pair:
                gray.append(cv2.cvtColor(side, cv2.COLOR_BGR2GRAY))
        else:
            gray = pair
        return self.block_matcher.compute(gray[0], gray[1])
    @property
    def search_range(self):
        """Number of disparities for ``block_matcher``."""
        return self._search_range
    @search_range.setter
    def search_range(self, value):
        """Set ``search_range`` to multiple of 16, replace ``block_matcher``."""
        if value == 0 or not value % 16:
            self._search_range = value
            self.replace_block_matcher()
        else:
            raise InvalidSearchRange("Search range must be a multiple of 16.")
    @property
    def window_size(self):
        """Search window size."""
        return self._window_size
    @window_size.setter
    def window_size(self, value):
        """Set search window size and update ``block_matcher``."""
        if value > 4 and value < 22 and value % 2:
            self._window_size = value
            self.replace_block_matcher()
        else:
            raise InvalidWindowSize("Window size must be an odd number between "
                                    "5 and 21 (inclusive).")
    @property
    def stereo_bm_preset(self):
        """Stereo BM preset used by ``block_matcher``."""
        return self._bm_preset
    @stereo_bm_preset.setter
    def stereo_bm_preset(self, value):
        """Set stereo BM preset and update ``block_matcher``."""
        if value in (cv2.STEREO_BM_BASIC_PRESET,
                     cv2.STEREO_BM_FISH_EYE_PRESET,
                     cv2.STEREO_BM_NARROW_PRESET):
            self._bm_preset = value
            self.replace_block_matcher()
        else:
            raise InvalidBMPreset("Stereo BM preset must be defined as "
                                  "cv2.STEREO_BM_*_PRESET.")
    def replace_block_matcher(self):
        """Replace ``block_matcher`` with current values."""
        self.block_matcher = cv2.StereoBM(preset=self._bm_preset,
                                          ndisparities=self._search_range,
                                          SADWindowSize=self._window_size)

class StereoBMTuner(object):
    """
    A class for tuning Stereo BM settings.

    Display a normalized disparity picture from two pictures captured with a
    ``CalibratedPair`` and allow the user to manually tune the settings for the
    stereo block matcher.
    """
    #: Window to show results in
    window_name = "Stereo BM Tuner"
    def __init__(self, calibrated_pair, image_pair):
        """Initialize tuner with a ``CalibratedPair`` and tune given pair."""
        #: Calibrated stereo pair to find Stereo BM settings for
        self.calibrated_pair = calibrated_pair
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("cam_preset", self.window_name,
                           self.calibrated_pair.stereo_bm_preset, 3,
                           self.set_bm_preset)
        cv2.createTrackbar("ndis", self.window_name,
                           self.calibrated_pair.search_range, 160,
                           self.set_search_range)
        cv2.createTrackbar("winsize", self.window_name,
                           self.calibrated_pair.window_size, 21,
                           self.set_window_size)
        #: (left, right) image pair to find disparity between
        self.pair = image_pair
        self.tune_pair(image_pair)
    def set_bm_preset(self, preset):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.stereo_bm_preset = preset
        except InvalidBMPreset:
            return
        self.update_disparity_map()
    def set_search_range(self, search_range):
        """Set ``search_range`` and update disparity image."""
        try:
            self.calibrated_pair.search_range = search_range
        except InvalidSearchRange:
            return
        self.update_disparity_map()
    def set_window_size(self, window_size):
        """Set ``window_size`` and update disparity image."""
        try:
            self.calibrated_pair.window_size = window_size
        except InvalidWindowSize:
            return
        self.update_disparity_map()
    def update_disparity_map(self):
        """Update disparity map in GUI."""
        disparity = self.calibrated_pair.compute_disparity(self.pair)
        cv2.imshow(self.window_name, disparity / 255.)
        cv2.waitKey()
    def tune_pair(self, pair):
        """Tune a pair of images."""
        self.pair = pair
        self.update_disparity_map()

def main():
    """Let user tune all images in the input folder and report chosen values."""
    parser = argparse.ArgumentParser(description="Read images taken from a "
                                     "calibrated stereo pair, compute "
                                     "disparity maps from them and show them "
                                     "interactively to the user, allowing the "
                                     "user to tune the stereo block matcher "
                                     "settings in the GUI.")
    parser.add_argument("calibration_folder",
                        help="Directory where calibration files for the stereo "
                        "pair are stored.")
    parser.add_argument("image_folder",
                        help="Directory where input images are stored.")
    args = parser.parse_args()

    calibration = calibrate_stereo.StereoCalibration(
                                        input_folder=args.calibration_folder)
    input_files = find_files(args.image_folder)
    calibrated_pair = CalibratedPair(None, calibration)
    image_pair = [cv2.imread(image) for image in input_files[:2]]
    rectified_pair = calibration.rectify(image_pair)
    tuner = StereoBMTuner(calibrated_pair, rectified_pair)
    chosen_arguments = []
    while input_files:
        image_pair = [cv2.imread(image) for image in input_files[:2]]
        rectified_pair = calibration.rectify(image_pair)
        tuner.tune_pair(rectified_pair)
        chosen_arguments.append((calibrated_pair.stereo_bm_preset,
                                 calibrated_pair.search_range,
                                 calibrated_pair.window_size))
        input_files = input_files[2:]
    stereo_bm_presets, search_ranges, window_sizes = [], [], []
    for preset, search_range, size in chosen_arguments:
        stereo_bm_presets.append(preset)
        search_ranges.append(search_range)
        window_sizes.append(size)
    for name, values in (("Stereo BM presets", stereo_bm_presets),
                         ("Search ranges", search_ranges),
                         ("Window sizes", window_sizes)):
        report_variable(name, values)
        print()
        
if __name__ == "__main__":
    main()
