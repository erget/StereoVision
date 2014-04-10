#!/bin/env python
"""
Module for finding chessboards with a stereo rig.
"""

import os
import time
import webcams

import argparse
import cv2
import progressbar


class ChessboardFinder(webcams.StereoPair):
    """A ``StereoPair`` that can find chessboards."""

    def get_chessboard(self, columns, rows, show=False):
        """
        Take a picture with a chessboard visible in both captures.

        ``columns`` and ``rows`` should be the number of inside corners in the
        chessboard's columns and rows. ``show`` determines whether the frames
        are shown while the cameras search for a chessboard.
        """
        found_chessboard = [False, False]
        while not all(found_chessboard):
            frames = self.get_frames()
            if show:
                self.show_frames(1)
            for i, frame in enumerate(frames):
                (found_chessboard[i],
                corners) = cv2.findChessboardCorners(frame, (columns, rows),
                                                  flags=cv2.CALIB_CB_FAST_CHECK)
        return frames

PROGRAM_DESCRIPTION=(
"Take a number of pictures with a stereo camera in which a chessboard is "
"visible to both cameras. The program waits until a chessboard is detected in "
"both camera frames. The pictures are then saved to a file in the specified "
"output folder. After five seconds, the cameras are rescanned to find another "
"chessboard perspective. This continues until the specified number of pictures "
"has been taken."
)

CHESSBOARD_ARGUMENTS = argparse.ArgumentParser(add_help=False)
CHESSBOARD_ARGUMENTS.add_argument("--rows", type=int,
                                  help="Number of inside corners in the "
                                  "chessboard's rows.", default=9)
CHESSBOARD_ARGUMENTS.add_argument("--columns", type=int,
                                  help="Number of inside corners in the "
                                  "chessboard's columns.", default=6)

def main():
    """
    Take a pictures with chessboard visible to both cameras in a stereo pair.
    """
    parser = argparse.ArgumentParser(description=PROGRAM_DESCRIPTION,
                                     parents=[CHESSBOARD_ARGUMENTS])
    parser.add_argument("left", metavar="left", type=int,
                        help="Device numbers for the left camera.")
    parser.add_argument("right", metavar="right", type=int,
                        help="Device numbers for the right camera.")
    parser.add_argument("num_pictures", type=int, help="Number of valid "
                        "chessboard pictures that should be taken.")
    parser.add_argument("output_folder", help="Folder to save the results to.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    with ChessboardFinder((args.left, args.right)) as pair:
        for i in range(args.num_pictures):
            print("Capturing picture {}...".format(i + 1))
            frames = pair.get_chessboard(args.columns, args.rows)
            for side, frame in zip(("left", "right"), frames):
                number_string = str(i + 1).zfill(len(str(args.num_pictures)))
                filename = "{}_{}.ppm".format(side, number_string)
                output_path = os.path.join(args.output_folder, filename)
                cv2.imwrite(output_path, frame)
            time.sleep(5)

if __name__ == "__main__":
    main()
