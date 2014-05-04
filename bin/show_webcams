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

import argparse
import os
import time

import cv2

from stereovision.stereo_cameras import StereoPair


def main():
    """
    Show the video from two webcams successively.

    For best results, connect the webcams while starting the computer.
    I have noticed that in some cases, if the webcam is not already connected
    when the computer starts, the USB device runs out of memory. Switching the
    camera to another USB port has also caused this problem in my experience.
    """
    parser = argparse.ArgumentParser(description="Show video from two "
                                     "webcams.\n\nPress 'q' to exit.")
    parser.add_argument("devices", type=int, nargs=2, help="Device numbers "
                        "for the cameras that should be accessed in order "
                        " (left, right).")
    parser.add_argument("--output_folder",
                        help="Folder to write output images to.")
    parser.add_argument("--interval", type=float, default=1,
                        help="Interval (s) to take pictures in.")
    args = parser.parse_args()

    with StereoPair(args.devices) as pair:
        if not args.output_folder:
            pair.show_videos()
        else:
            i = 1
            while True:
                start = time.time()
                while time.time() < start + args.interval:
                    pair.show_frames(1)
                images = pair.get_frames()
                for side, image in zip(("left", "right"), images):
                    filename = "{}_{}.ppm".format(side, i)
                    output_path = os.path.join(args.output_folder, filename)
                    cv2.imwrite(output_path, image)
                i += 1


if __name__ == "__main__":
    main()
