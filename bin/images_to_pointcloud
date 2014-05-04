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
Tool for creating and exporting colored point clouds from stereo image pairs.
"""

import argparse

import cv2
from stereovision.blockmatchers import StereoBM, StereoSGBM
from stereovision.calibration import StereoCalibration
from stereovision.stereo_cameras import CalibratedPair
from stereovision.ui_utils import STEREO_BM_FLAG


def main():
    """Produce PLY point clouds from stereo image pair."""
    parser = argparse.ArgumentParser(description="Read images taken with "
                                     "stereo pair and use them to produce 3D "
                                     "point clouds that can be viewed with "
                                     "MeshLab.", parents=[STEREO_BM_FLAG])
    parser.add_argument("calibration", help="Path to calibration folder.")
    parser.add_argument("left", help="Path to left image")
    parser.add_argument("right", help="Path to right image")
    parser.add_argument("output", help="Path to output file.")
    parser.add_argument("--bm_settings",
                        help="Path to block matcher's settings.")
    args = parser.parse_args()

    image_pair = [cv2.imread(image) for image in [args.left, args.right]]
    calib_folder = args.calibration
    if args.use_stereobm:
        block_matcher = StereoBM()
    else:
        block_matcher = StereoSGBM()
    if args.bm_settings:
        block_matcher.load_settings(args.bm_settings)

    camera_pair = CalibratedPair(None,
                                StereoCalibration(input_folder=calib_folder),
                                block_matcher)
    rectified_pair = camera_pair.calibration.rectify(image_pair)
    points = camera_pair.get_point_cloud(rectified_pair)
    points = points.filter_infinity()
    points.write_ply(args.output)


if __name__ == "__main__":
    main()
