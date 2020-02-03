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
import time

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

        # Sets initial position of windows, based on image size
        set_window_position(pair)

        for i in range(args.num_pictures):

            # Introduces a 5 second delay before the camera pair is scanned for new images
            enforce_delay(pair, 5)

            frames, corners = pair.get_chessboard(args.columns, args.rows, True)
            for side, frame in zip(("left", "right"), frames):
                number_string = str(i + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)

            progress.update(progress.maxval - (args.num_pictures - i))

            # Displays the recent accepted image pair. Helps in generating diverse calibration images.
            show_selected_frames(frames, corners, pair, args, True)

        progress.finish()
        cv2.destroyAllWindows()

    if args.calibration_folder:
        args.input_files = find_files(args.output_folder)
        args.output_folder = args.calibration_folder
        args.show_chessboards = True
        calibrate_folder(args)


def show_selected_frames(frames, corners, pair, args, draw_corners=False):
    """
    Display the most recently captured (left as well as right) images.
    If draw_corners is set to true, the identified corners are marked on the images.
    """

    if draw_corners:
        for frame, corner in zip(frames, corners):
            cv2.drawChessboardCorners(frame, (args.columns, args.rows), corner, True)

    cv2.imshow("{} selected".format(pair.windows[0]), frames[0])
    cv2.imshow("{} selected".format(pair.windows[1]), frames[1])


def enforce_delay(pair, delay):
    """
    Enforces a delay of 5 seconds. This helps the user to change the chessboard perspective.
    A timer is displayed indicating the time remaining before the next sample is captured.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_4
    line_thickness = 4

    start_time = time.time()
    now = start_time

    while now - start_time < delay:

        frames = pair.get_frames()

        # Calculates the time remaining before the next sample is captured
        time_remaining = "{:.2f}".format(delay - now + start_time)

        # Estimating the scale factor.
        font_scale = get_approx_font_scale(frames[0], time_remaining, font, line_thickness)

        text_size = cv2.getTextSize(time_remaining, font, font_scale, line_thickness)[0]

        # Calculates the position of the text
        text_x = (frames[0].shape[1] - text_size[0]) / 2
        text_y = (frames[0].shape[0] + text_size[1]) / 2

        for frame, window in zip(frames, pair.windows):
            cv2.putText(frame, time_remaining, (text_x, text_y), font, font_scale, (255, 50, 50),
                        line_thickness, line_type)
            cv2.imshow(window, frame)

        cv2.waitKey(1)
        now = time.time()


def get_approx_font_scale(frame, text, font, line_thickness):
    """
    Approximate the font scale for the timer display.
    """

    _, width = frame.shape[:2]
    target_width = width / 2

    base_text_size = cv2.getTextSize(text, font, 1.0, line_thickness)[0]
    scale_factor = float(target_width) / base_text_size[0]

    return scale_factor


def set_window_position(pair):

    """
    Set initial the positions of windows.
    The top left and right windows display the live cam stream with timer overlay.
    The bottom left and right windows display recently selected frame.
    """

    frames = pair.get_frames()
    pair.show_frames(1)

    # Setting initial position of cameras
    cv2.moveWindow(pair.windows[0], 0, 0)
    cv2.moveWindow(pair.windows[1], frames[1].shape[1], 0)

    # Setting initial position of selected frames
    cv2.namedWindow("{} selected".format(pair.windows[0]), cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("{} selected".format(pair.windows[0]), 0, frames[0].shape[0] + 30)

    cv2.namedWindow("{} selected".format(pair.windows[1]), cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow("{} selected".format(pair.windows[1]), frames[1].shape[1], frames[1].shape[0] + 30)


if __name__ == "__main__":
    main()
