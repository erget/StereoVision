#!/bin/env python
'''
class for working with calibrated stereo camera pairs.
'''

import argparse
from calibrate_stereo import find_files, show_image
import calibrate_stereo
import point_cloud
import webcams

import cv2
import simplejson

def report_variable(variable_name, variable):
    """
    Check how often each parameter value was chosen.

    Return most common setting and report over all settings chosen.
    """
    report = []
    unique_values = list(set(variable))
    value_frequency = {}
    for value in unique_values:
        value_frequency[variable.count(value)] = value
    frequencies = value_frequency.keys()
    frequencies.sort(reverse=True)
    header = "{} value | Selection frequency".format(variable_name)
    left_column_width = len(header[:-21])
    right_column_width = 21
    report.append(header)
    report.append("{}|{}".format("-" * left_column_width, "-" *
                                 right_column_width))
    for frequency in frequencies:
        left_column = str(value_frequency[frequency]).center(left_column_width)
        right_column = str(frequency).center(right_column_width)
        report.append("{}|{}".format(left_column, right_column))
    return value_frequency[frequencies[0]], "\n".join(report + ["\n"])

class CalibratedPair(webcams.StereoPair):
    """
    A stereo pair of calibrated cameras.

    Should be initialized with a context manager to ensure that the camera
    connections are closed properly.
    """
    def __init__(self, devices, calibration, block_matcher):
        """
        Initialize cameras.

        ``devices`` is an iterable of the device numbers. If you want to use the
        ``CalibratedPair`` in offline mode, pass None.
        ``calibration`` is a StereoCalibration object.
        ``block_matcher`` is a BlockMatcher object.
        """
        if devices:
            super(CalibratedPair, self).__init__(devices)
        #: ``StereoCalibration`` object holding the camera pair's calibration
        self.calibration = calibration
        #: ``BlockMatcher`` object for computing disparity and point cloud
        self.block_matcher = block_matcher
    def get_frames(self):
        """Rectify and return current frames from cameras."""
        frames = super(CalibratedPair, self).get_frames()
        return self.calibration.rectify(frames)
    def get_point_cloud(self, pair):
        """Get 3D point cloud from image pair."""
        disparity = self.block_matcher.compute_disparity(pair)
        points = self.block_matcher.compute_3d(disparity,
                                           self.calibration.disp_to_depth_mat)
        colors = cv2.cvtColor(pair[0], cv2.COLOR_BGR2RGB)
        return point_cloud.PointCloud(points, colors)


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
        """
        Initialize tuner with a ``CalibratedPair`` and tune given pair.

        ``ndis`` is limited to the number of pixels equal to or less than an
        individual image's shortest dimension that is evenly divisible by 16.
        """
        #: Calibrated stereo pair to find Stereo BM settings for
        self.calibrated_pair = calibrated_pair
        #: (left, right) image pair to find disparity between
        self.pair = image_pair
        cv2.namedWindow(self.window_name)
        shortest_dimension = min(self.pair[0].shape[:2])
        cv2.createTrackbar("cam_preset", self.window_name,
                           self.calibrated_pair.stereo_bm_preset, 3,
                           self.set_bm_preset)
        cv2.createTrackbar("ndis", self.window_name,
                           self.calibrated_pair.search_range,
                           shortest_dimension / 16 * 16,
                           self.set_search_range)
        cv2.createTrackbar("winsize", self.window_name,
                           self.calibrated_pair.window_size,
                           self.calibrated_pair.max_winsize,
                           self.set_window_size)
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
        """
        Update disparity map in GUI.

        The disparity image is normalized to the range 0-255 and then divided by
        255, because OpenCV multiplies it by 255 when displaying. This is
        because the pixels are stored as floating points.
        """
        disparity = self.calibrated_pair.compute_disparity(self.pair)
        cv2.imshow(self.window_name, disparity * 255)
        cv2.waitKey()
    def tune_pair(self, pair):
        """Tune a pair of images."""
        self.pair = pair
        self.update_disparity_map()

def main():
    """
    Let user tune all images in the input folder and report chosen values.

    Load all images from input folder, consuming available files and showing
    them in the GUI. Afterwards, report user's chosen settings and, if a file
    for the BM settings is provided, save the most common settings to file.
    """
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
    parser.add_argument("--bm_settings",
                        help="File to save most commonly chosen block matcher "
                        "settings to.", default="")
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
    pretty_settings_names = {"stereo_bm_preset": "Stereo BM presets",
                             "search_range": "Search ranges",
                             "window_size": "Window sizes"}
    common_settings = {}
    for variable, values in (("stereo_bm_preset", stereo_bm_presets),
                         ("search_range", search_ranges),
                         ("window_size", window_sizes)):
        (common_setting,
         report) = report_variable(pretty_settings_names[variable], values)
        common_settings[variable] = common_setting
        print(report)
    if args.bm_settings:
        with open(args.bm_settings, "w") as settings_file:
            simplejson.dump(common_settings, settings_file)

if __name__ == "__main__":
    main()
