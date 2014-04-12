#!/bin/env python
"""
Module for calibrating stereo cameras from a images with chessboards.

See stereo_match.tune_calib_pair in the OpenCV examples for some additional tips on
converting the generated 3D coordinates into a point cloud. I borrowed some
numpy magic from there.
"""

import argparse
import os

import cv2
import progressbar

import numpy as np


class ChessboardNotFoundError(Exception):
    """No chessboard could be found in searched image."""


def show_image(image, window_name, wait=0):
    """Show an image and exit when a key is pressed or specified wait period."""
    cv2.imshow(window_name, image)
    if cv2.waitKey(wait):
        cv2.destroyWindow(window_name)


def find_files(folder):
    """Discover stereo photos and return them as a pairwise sorted list."""
    files = [i for i in os.listdir(folder) if i.startswith("left")]
    files.sort()
    for i in range(len(files)):
        insert_string = "right{}".format(files[i * 2][4:])
        files.insert(i * 2 + 1, insert_string)
    files = [os.path.join(folder, filename) for filename in files]
    return files


class StereoCalibration(object):
    """
    Camera calibration.

    A ``StereoCalibration`` can be used to store the calibration for a stereo
    pair from a collection of pictures. With this calibration, it can also
    rectify pictures taken from its stereo pair.
    """
    def __init__(self, calibration=None, input_folder=None):
        """
        Initialize camera calibration.

        If another calibration object is provided, copy its values. If an input
        folder is provided, load *.npy files from that folder. An input folder
        overwrites a calibration object.
        """
        #: Camera matrices
        self.cam_mats = {"left": None, "right": None}
        #: Distortion coefficients
        self.dist_coefs = {"left": None, "right": None}
        #: Rotation matrix
        self.rot_mat = None
        #: Translation vector
        self.trans_vec = None
        #: Essential matrix
        self.e_mat = None
        #: Fundamental matrix
        self.f_mat = None
        #: Rectification transforms
        self.rect_trans = {"left": None, "right": None}
        #: Projection matrices
        self.proj_mats = {"left": None, "right": None}
        #: Disparity to depth mapping matrix
        self.disp_to_depth_mat = None
        #: Bounding boxes of valid pixels
        self.valid_boxes = {"left": None, "right": None}
        #: Undistortion maps for remapping
        self.undistortion_map = {"left": None, "right": None}
        #: Rectification maps for remapping
        self.rectification_map = {"left": None, "right": None}
        if calibration:
            self.copy_calibration(calibration)
        elif input_folder:
            self.load(input_folder)
    def __str__(self):
        output = ""
        for key, item in self.__dict__.items():
            output += key + ":\n"
            output += str(item) + "\n"
        return output
    def copy_calibration(self, calibration):
        """Copy another ``StereoCalibration`` object's values."""
        for key, item in calibration.__dict__.items():
            self.__dict__[key] = item
    def _interact_with_folder(self, output_folder, action):
        """
        Export/import matrices as *.npy files to/from an output folder.

        ``action`` is a string. It determines whether the method reads or writes
        to disk. It must have one of the following values: ('r', 'w').
        """
        if not action in ('r', 'w'):
            raise ValueError("action must be either 'r' or 'w'.")
        for key, item in self.__dict__.items():
            if isinstance(item, dict):
                for side in ("left", "right"):
                    filename = os.path.join(output_folder,
                                            "{}_{}.npy".format(key, side))
                    if action == 'w':
                        np.save(filename, self.__dict__[key][side])
                    else:
                        self.__dict__[key][side] = np.load(filename)
            else:
                filename = os.path.join(output_folder, "{}.npy".format(key))
                if action == 'w':
                    np.save(filename, self.__dict__[key])
                else:
                    self.__dict__[key] = np.load(filename)
    def export(self, output_folder):
        """Export matrices as *.npy files to an output folder."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self._interact_with_folder(output_folder, 'w')
    def load(self, input_folder):
        """Load values from *.npy files in ``input_folder``."""
        self._interact_with_folder(input_folder, 'r')
    def rectify(self, frames):
        """
        Rectify frames passed as (left, right) pair of OpenCV Mats.

        Remapping is done with nearest neighbor for speed.
        """
        new_frames = []
        for i, side in enumerate(("left", "right")):
            new_frames.append(cv2.remap(frames[i],
                                        self.undistortion_map[side],
                                        self.rectification_map[side],
                                        cv2.INTER_NEAREST))
        return new_frames


class StereoCalibrator(object):
    """A class that calibrates stereo cameras."""
    def __init__(self, rows, columns, square_size, image_size):
        """
        Store variables relevant to the camera calibration.

        ``corner_coordinates`` are generated by creating an array of 3D
        coordinates that correspond to the actual positions of the chessboard
        corners observed on a 2D plane in 3D space.
        """
        #: Number of calibration images
        self.image_count = 0
        #: Number of inside corners in the chessboard's rows
        self.rows = rows
        #: Number of inside corners in the chessboard's columns
        self.columns = columns
        #: Size of chessboard squares in cm
        self.square_size = square_size
        #: Size of calibration images in pixels
        self.image_size = image_size
        pattern_size = (self.rows, self.columns)
        corner_coordinates = np.zeros((np.prod(pattern_size), 3), np.float32)
        corner_coordinates[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        corner_coordinates *= self.square_size
        #: Real world corner coordinates found in each image
        self.corner_coordinates = corner_coordinates
        #: Array of real world corner coordinates to match the corners found
        self.object_points = []
        #: Array of found corner coordinates from calibration images for left
        #: and right camera, respectively
        self.image_points = {"left": [], "right": []}
    def get_corners(self, image):
        """Find subpixel chessboard corners in image."""
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(temp,
                                                 (self.rows, self.columns))
        if not ret:
            raise ChessboardNotFoundError("No chessboard could be found.")
        cv2.cornerSubPix(temp, corners, (11, 11), (-1, -1),
                         (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                          30, 0.01))
        return corners
    def show_corners(self, image, corners):
        """Show chessboard corners found in image."""
        temp = image
        cv2.drawChessboardCorners(temp, (self.rows, self.columns), corners,
                                  True)
        show_image(temp, "Chessboard")
    def add_corners(self, image_pair, show_results=False):
        """
        Record chessboard corners found in an image pair.

        The image pair should be an iterable composed of two CvMats ordered
        (left, right).
        """
        side = "left"
        self.object_points.append(self.corner_coordinates)
        for image in image_pair:
            corners = self.get_corners(image)
            if show_results:
                self.show_corners(image, corners)
            self.image_points[side].append(corners.reshape(-1, 2))
            side = "right"
            self.image_count += 1
    def calibrate_cameras(self):
        """Calibrate cameras based on found chessboard corners."""
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    100, 1e-5)
        flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
                 cv2.CALIB_SAME_FOCAL_LENGTH)
        calib = StereoCalibration()
        (calib.cam_mats["left"], calib.dist_coefs["left"],
         calib.cam_mats["right"], calib.dist_coefs["right"],
         calib.rot_mat, calib.trans_vec, calib.e_mat,
         calib.f_mat) = cv2.stereoCalibrate(self.object_points,
                                            self.image_points["left"],
                                            self.image_points["right"],
                                            self.image_size,
                                            criteria=criteria,
                                            flags=flags)[1:]
        (calib.rect_trans["left"], calib.rect_trans["right"],
         calib.proj_mats["left"], calib.proj_mats["right"],
         calib.disp_to_depth_mat, calib.valid_boxes["left"],
         calib.valid_boxes["right"]) = cv2.stereoRectify(calib.cam_mats["left"],
                                                      calib.dist_coefs["left"],
                                                      calib.cam_mats["right"],
                                                      calib.dist_coefs["right"],
                                                      self.image_size,
                                                      calib.rot_mat,
                                                      calib.trans_vec,
                                                      flags=0)
        for side in ("left", "right"):
            (calib.undistortion_map[side],
             calib.rectification_map[side]) = cv2.initUndistortRectifyMap(
                                                        calib.cam_mats[side],
                                                        calib.dist_coefs[side],
                                                        calib.rect_trans[side],
                                                        calib.proj_mats[side],
                                                        self.image_size,
                                                        cv2.CV_32FC1)
        return calib
    def check_calibration(self, calibration):
        """
        Check calibration quality by computing average reprojection error.

        First, undistort detected points and compute epilines for each side.
        Then compute the error between the computed epipolar lines and the
        position of the points detected on the other side for each point and
        return the average error.
        """
        sides = "left", "right"
        which_image = {sides[0]: 1, sides[1]: 2}
        undistorted, lines = {}, {}
        for side in sides:
            undistorted[side] = cv2.undistortPoints(
                         np.concatenate(self.image_points[side]).reshape(-1,
                                                                         1, 2),
                         calibration.cam_mats[side],
                         calibration.dist_coefs[side],
                         P=calibration.cam_mats[side])
            lines[side] = cv2.computeCorrespondEpilines(undistorted[side],
                                              which_image[side],
                                              calibration.f_mat)
        total_error = 0
        this_side, other_side = sides
        for side in sides:
            for i in range(len(undistorted[side])):
                total_error += abs(undistorted[this_side][i][0][0] *
                                   lines[other_side][i][0][0] +
                                   undistorted[this_side][i][0][1] *
                                   lines[other_side][i][0][1] +
                                   lines[other_side][i][0][2])
            other_side, this_side = sides
        total_points = self.image_count * len(self.object_points)
        return total_error / total_points

def calibrate_folder(args):
    """
    Calibrate camera based on chessboard images, write results to output folder.

    All images are read from disk. Chessboard points are found and used to
    calibrate the stereo pair. Finally, the calibration is written to the folder
    specified in ``args``.

    ``args`` needs to contain the following fields:
        input_files: List of paths to input files
        rows: Number of rows in chessboard
        columns: Number of columns in chessboard
        square_size: Size of chessboard squares in cm
        output_folder: Folder to write calibration to
    """
    height, width = cv2.imread(args.input_files[0]).shape[:2]
    calibrator = StereoCalibrator(args.rows, args.columns, args.square_size,
                                  (width, height))
    progress = progressbar.ProgressBar(maxval=len(args.input_files),
                                       widgets=[progressbar.Bar("=", "[", "]"),
                                                " ", progressbar.Percentage()])
    print("Reading input files...")
    while args.input_files:
        left, right = args.input_files[:2]
        img_left, im_right = cv2.imread(left), cv2.imread(right)
        calibrator.add_corners((img_left, im_right),
                               show_results=args.show_chessboards)
        args.input_files = args.input_files[2:]
        progress.update(progress.maxval - len(args.input_files))

    progress.finish()
    print("Calibrating cameras. This can take a while.")
    calibration = calibrator.calibrate_cameras()
    avg_error = calibrator.check_calibration(calibration)
    print("The average error between chessboard points and their epipolar "
          "lines is \n"
          "{} pixels. This should be as small as possible.".format(avg_error))
    calibration.export(args.output_folder)

CHESSBOARD_ARGUMENTS = argparse.ArgumentParser(add_help=False)
CHESSBOARD_ARGUMENTS.add_argument("--rows", type=int,
                                  help="Number of inside corners in the "
                                  "chessboard's rows.", default=9)
CHESSBOARD_ARGUMENTS.add_argument("--columns", type=int,
                                  help="Number of inside corners in the "
                                  "chessboard's columns.", default=6)
CHESSBOARD_ARGUMENTS.add_argument("--square-size", help="Size of chessboard "
                                  "squares in cm.", type=float, default=1.8)

def main():
    """
    Read all images in input folder and produce camera calibration files.

    First, parse arguments provided by user. Then scan input folder for input
    files. Harvest chessboard points from each image in folder, then use them
    to calibrate the stereo pair. Report average error to user and export
    calibration files to output folder.
    """
    parser = argparse.ArgumentParser(description="Read images taken with "
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

    args.input_files = find_files(args.input_folder)
    calibrate_folder(args)

if __name__ == "__main__":
    main()
