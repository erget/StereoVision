#!/bin/python
## Copyright (C) 2014 Daniel Lee <lee.daniel.1986@gmail.com>
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
Calibrate stereo camera based on detected chessboard corners.
"""

from argparse import ArgumentParser
from stereovision.ui_utils import (find_files, calibrate_folder,
                                  CHESSBOARD_ARGUMENTS)


def main():
    """
    Read all images in input folder and produce camera calibration files.

    First, parse arguments provided by user. Then scan input folder for input
    files. Harvest chessboard points from each image in folder, then use them
    to calibrate the stereo pair. Report average error to user and export
    calibration files to output folder.
    """
    parser = ArgumentParser(description="Read images taken with "
                                     "stereo pair and use them to compute "
                                     "camera calibration.",
                             parents=[CHESSBOARD_ARGUMENTS])
    parser.add_argument("input_folder", help="Input folder assumed to contain "
                        "only stereo images taken with the stereo camera pair "
                        "that should be calibrated.")
    parser.add_argument("output_folder", help="Folder to write calibration "
                        "files to.", default="/tmp/")
    parser.add_argument("--show-chessboards", help="Display detected "
                        "chessboard corners.", action="store_true")
    args = parser.parse_args()

    print args.input_folder
    args.input_files = find_files(args.input_folder)
    calibrate_folder(args)


if __name__ == "__main__":
    main()
