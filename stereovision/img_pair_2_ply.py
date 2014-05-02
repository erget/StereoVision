#!/bin/env python
"""
Tool for creating and exporting colored point clouds from stereo image pairs.
"""

import argparse
from blockmatchers import StereoBM, StereoSGBM
from calibrate_stereo import find_files, StereoCalibration
from tune_calib_pair import CalibratedPair, STEREO_SGBM_FLAG

import cv2


def main():
    """Produce PLY point clouds from stereo image pair."""
    parser = argparse.ArgumentParser(description="Read images taken with "
                                     "stereo pair and use them to produce 3D "
                                     "point clouds that can be viewed with "
                                     "MeshLab.", parents=[STEREO_SGBM_FLAG])
    parser.add_argument("calibration", help="Path to calibration folder.")
    parser.add_argument("left", help="Path to left image")
    parser.add_argument("right", help="Path to right image")
    parser.add_argument("output", help="Path to output file.")
    parser.add_argument("--bm_settings", help="Path to block matcher's settings.")
    args = parser.parse_args()

    image_pair = [cv2.imread(image) for image in [args.left, args.right]]
    calibration_folder = args.calibration
    if args.use_stereobm:
        block_matcher = StereoBM()
    else:
        block_matcher = StereoSGBM()
    if args.bm_settings:
        block_matcher.load_settings(args.bm_settings)

    cp = CalibratedPair(None, StereoCalibration(input_folder=calibration_folder),
                        block_matcher)
    rectified_pair = cp.calibration.rectify(image_pair)
    points = cp.get_point_cloud(rectified_pair)
    points = points.filter_infinity()
    points.write_ply(args.output)

if __name__ == "__main__":
    main()
