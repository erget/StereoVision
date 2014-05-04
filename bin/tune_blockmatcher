#!/bin/python
# Copyright (C) 2014 Daniel Lee <lee.daniel.1986@gmail.com>
#
# This file is part of StereoVision.
#
# StereoVision is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# StereoVision is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with StereoVision.  If not, see <http://www.gnu.org/licenses/>.

"""
Let user tune all images in the input folder and report chosen values.

Load a calibration from file and instantiate a ``BlockMatcher`` of the type
requested by the user. Load images successively from input folder and display
their resultant disparity map generated with the ``BlockMatcher`` and the
parameters chosen in the ``BMTuner``'s GUI. Afterwards, report user's chosen
settings and, if a file for the BM settings is provided, save the most common
settings to file.
"""

from argparse import ArgumentParser

import cv2

from stereovision.blockmatchers import StereoBM, StereoSGBM
from stereovision.calibration import StereoCalibration
from stereovision.ui_utils import find_files, BMTuner, STEREO_BM_FLAG


def main():
    parser = ArgumentParser(description="Read images taken from a calibrated "
                           "stereo pair, compute disparity maps from them and "
                           "show them interactively to the user, allowing the "
                           "user to tune the stereo block matcher settings in "
                           "the GUI.", parents=[STEREO_BM_FLAG])
    parser.add_argument("calibration_folder",
                        help="Directory where calibration files for the stereo "
                        "pair are stored.")
    parser.add_argument("image_folder",
                        help="Directory where input images are stored.")
    parser.add_argument("--bm_settings",
                        help="File to save last block matcher settings to.",
                        default="")
    args = parser.parse_args()

    calibration = StereoCalibration(input_folder=args.calibration_folder)
    input_files = find_files(args.image_folder)
    if args.use_stereobm:
        block_matcher = StereoBM()
    else:
        block_matcher = StereoSGBM()
    image_pair = [cv2.imread(image) for image in input_files[:2]]
    input_files = input_files[2:]
    rectified_pair = calibration.rectify(image_pair)
    tuner = BMTuner(block_matcher, calibration, rectified_pair)

    while input_files:
        image_pair = [cv2.imread(image) for image in input_files[:2]]
        rectified_pair = calibration.rectify(image_pair)
        tuner.tune_pair(rectified_pair)
        input_files = input_files[2:]

    for param in block_matcher.parameter_maxima:
        print("{}\n".format(tuner.report_settings(param)))

    if args.bm_settings:
        block_matcher.save_settings(args.bm_settings)


if __name__ == "__main__":
    main()
