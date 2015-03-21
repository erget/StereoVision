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
Take pictures of a chessboard visible to both cameras in a stereo pair.
"""

from argparse import ArgumentParser
import os

import cv2
from progressbar import ProgressBar, Bar, Percentage
from stereovision.stereo_cameras import ChessboardFinder
from stereovision.ui_utils import calibrate_folder, CHESSBOARD_ARGUMENTS
from stereovision.ui_utils import find_files


PROGRAM_DESCRIPTION=(
"Take a number of pictures with a stereo camera in which a chessboard is "
"visible to both cameras. The program waits until a chessboard is detected in "
"both camera frames. The pictures are then saved to a file in the specified "
"output folder. After five seconds, the cameras are rescanned to find another "
"chessboard perspective. This continues until the specified number of pictures "
"has been taken."
)


def main():
    parser = ArgumentParser(description=PROGRAM_DESCRIPTION,
                           parents=[CHESSBOARD_ARGUMENTS])
    parser.add_argument("left", metavar="left", type=int,
                        help="Device numbers for the left camera.")
    parser.add_argument("right", metavar="right", type=int,
                        help="Device numbers for the right camera.")
    parser.add_argument("num_pictures", type=int, help="Number of valid "
                        "chessboard pictures that should be taken.")
    parser.add_argument("output_folder", help="Folder to save the images to.")
    parser.add_argument("--calibration-folder", help="Folder to save camera "
                        "calibration to.")
    args = parser.parse_args()
    if args.calibration_folder and not args.square_size:
        args.print_help()

    progress = ProgressBar(maxval=args.num_pictures,
                          widgets=[Bar("=", "[", "]"),
                          " ", Percentage()])
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    progress.start()
    with ChessboardFinder((args.left, args.right)) as pair:
        for i in range(args.num_pictures):
            frames = pair.get_chessboard(args.columns, args.rows, True)
            for side, frame in zip(("left", "right"), frames):
                number_string = str(i + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)
            progress.update(progress.maxval - (args.num_pictures - i))
            for i in range(10):
                pair.show_frames(1)
        progress.finish()
    if args.calibration_folder:
        args.input_files = find_files(args.output_folder)
        args.output_folder = args.calibration_folder
        args.show_chessboards = True
        calibrate_folder(args)


if __name__ == "__main__":
    main()
