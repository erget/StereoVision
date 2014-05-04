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

'''
Wrapper classes for block matching algorithms.

Classes:

    * ``BlockMatcher`` - Abstract class that implements interface for subclasses

        * ``StereoBM`` - StereoBM block matching algorithm
        * ``StereoSGBM`` - StereoSGBM block matching algorithm

.. image:: classes_blockmatchers.svg
'''

import cv2
import simplejson

import numpy as np
from stereovision.exceptions import (InvalidSearchRangeError,
                                    InvalidWindowSizeError,
                                    InvalidBMPresetError,
                                    InvalidNumDisparitiesError,
                                    InvalidSADWindowSizeError,
                                    InvalidUniquenessRatioError,
                                    InvalidSpeckleWindowSizeError,
                                    InvalidSpeckleRangeError,
                                    InvalidFirstDisparityChangePenaltyError,
                                    InvalidSecondDisparityChangePenaltyError)


class BlockMatcher(object):

    """
    Block matching algorithms.

    This abstract class exposes the interface for subclasses that wrap OpenCV's
    block matching algorithms. Doing so makes it possible to use them in the
    strategy pattern. In this library, that happens in ``CalibratedPair``, which
    uses a unified interface to interact with any kind of block matcher, and
    with ``BMTuners``, which can discover the ``BlockMatcher's`` parameters and
    allow the user to adjust them online.

    Each ``BlockMatcher`` protects its block matcher's parameters by using
    getters and setters. It exposes its settable parameter and their maximum
    values, if they exist, in the dictionary ``parameter_maxima``.

    ``load_settings``, ``save_settings`` and ``get_3d`` are implemented on
    ``BlockMatcher`` itself, as these are independent of the block matching
    algorithm. Subclasses are expected to implement ``_replace_bm`` and
    ``get_disparity``, as well as the getters and setters. They are also
    expected to call ``BlockMatcher``'s ``__init__`` after setting their own
    private variables.
    """

    #: Dictionary of parameter names associated with their maximum values
    parameter_maxima = {}

    def __init__(self, settings=None):
        """Set block matcher parameters and load from file if necessary."""
        #: Block matcher object used for computing point clouds
        self._block_matcher = None
        self._replace_bm()
        if settings:
            self.load_settings(settings)

    def load_settings(self, settings):
        """Load settings from file"""
        with open(settings) as settings_file:
            settings_dict = simplejson.load(settings_file)
        for key, value in settings_dict.items():
            self.__setattr__(key, value)

    def save_settings(self, settings_file):
        """Save block matcher settings to a file object"""
        settings = {}
        for parameter in self.parameter_maxima:
            settings[parameter] = self.__getattribute__(parameter)
        with open(settings_file, "w") as settings_file:
            simplejson.dump(settings, settings_file)

    @classmethod
    def get_3d(cls, disparity, disparity_to_depth_map):
        """Compute point cloud."""
        return cv2.reprojectImageTo3D(disparity, disparity_to_depth_map)

    def _replace_bm(self):
        """Replace block matcher with new parameters"""
        raise NotImplementedError

    def get_disparity(self, image_pair):
        """Compute disparity map from image pair."""
        raise NotImplementedError


class StereoBM(BlockMatcher):

    """A stereo block matching ``BlockMatcher``."""

    parameter_maxima = {"search_range": None,
                       "window_size": 255,
                       "stereo_bm_preset": cv2.STEREO_BM_NARROW_PRESET}

    @property
    def search_range(self):
        """Return private ``_search_range`` value."""
        return self._search_range

    @search_range.setter
    def search_range(self, value):
        """Set private ``_search_range`` and reset ``_block_matcher``."""
        if value == 0 or not value % 16:
            self._search_range = value
        else:
            raise InvalidSearchRangeError("Search range must be a multiple of "
                                          "16.")
        self._replace_bm()

    @property
    def window_size(self):
        """Return private ``_window_size`` value."""
        return self._window_size

    @window_size.setter
    def window_size(self, value):
        """Set private ``_window_size`` and reset ``_block_matcher``."""
        if (value > 4 and
            value < self.parameter_maxima["window_size"] and
            value % 2):
            self._window_size = value
        else:
            raise InvalidWindowSizeError("Window size must be an odd number "
                                      "between 0 and {}.".format(
                                      self.parameter_maxima["window_size"] + 1))
        self._replace_bm()

    @property
    def stereo_bm_preset(self):
        """Return private ``_bm_preset`` value."""
        return self._bm_preset

    @stereo_bm_preset.setter
    def stereo_bm_preset(self, value):
        """Set private ``_stereo_bm_preset`` and reset ``_block_matcher``."""
        if value in (cv2.STEREO_BM_BASIC_PRESET,
                     cv2.STEREO_BM_FISH_EYE_PRESET,
                     cv2.STEREO_BM_NARROW_PRESET):
            self._bm_preset = value
        else:
            raise InvalidBMPresetError("Stereo BM preset must be defined as "
                                       "cv2.STEREO_BM_*_PRESET.")
        self._replace_bm()

    def _replace_bm(self):
        """Replace ``_block_matcher`` with current values."""
        self._block_matcher = cv2.StereoBM(preset=self._bm_preset,
                                          ndisparities=self._search_range,
                                          SADWindowSize=self._window_size)

    def __init__(self, stereo_bm_preset=cv2.STEREO_BM_BASIC_PRESET,
                 search_range=80,
                 window_size=21,
                 settings=None):
        self._bm_preset = cv2.STEREO_BM_BASIC_PRESET
        self._search_range = 0
        self._window_size = 5
        #: OpenCV camera type for ``_block_matcher``
        self.stereo_bm_preset = stereo_bm_preset
        #: Number of disparities for ``_block_matcher``
        self.search_range = search_range
        #: Search window size for ``_block_matcher``
        self.window_size = window_size
        super(StereoBM, self).__init__(settings)

    def get_disparity(self, pair):
        """
        Compute disparity from image pair (left, right).

        First, convert images to grayscale if needed. Then pass to the
        ``_block_matcher`` for stereo matching.
        """
        gray = []
        if pair[0].ndim == 3:
            for side in pair:
                gray.append(cv2.cvtColor(side, cv2.COLOR_BGR2GRAY))
        else:
            gray = pair
        return self._block_matcher.compute(gray[0], gray[1],
                                          disptype=cv2.CV_32F)


class StereoSGBM(BlockMatcher):

    """A semi-global block matcher."""

    parameter_maxima = {"minDisparity": None,
                       "numDisparities": None,
                       "SADWindowSize": 11,
                       "P1": None,
                       "P2": None,
                       "disp12MaxDiff": None,
                       "uniquenessRatio": 15,
                       "speckleWindowSize": 200,
                       "speckleRange": 2,
                       "fullDP": 1}

    @property
    def minDisparity(self):
        """Return private ``_min_disparity`` value."""
        return self._min_disparity

    @minDisparity.setter
    def minDisparity(self, value):
        """Set private ``_min_disparity`` and reset ``_block_matcher``."""
        self._min_disparity = value
        self._replace_bm()

    @property
    def numDisparities(self):
        """Return private ``_num_disp`` value."""
        return self._num_disp

    @numDisparities.setter
    def numDisparities(self, value):
        """Set private ``_num_disp`` and reset ``_block_matcher``."""
        if value > 0 and value % 16 == 0:
            self._num_disp = value
        else:
            raise InvalidNumDisparitiesError("numDisparities must be a "
                                             "positive integer evenly "
                                             "divisible by 16.")
        self._replace_bm()

    @property
    def SADWindowSize(self):
        """Return private ``_sad_window_size`` value."""
        return self._sad_window_size

    @SADWindowSize.setter
    def SADWindowSize(self, value):
        """Set private ``_sad_window_size`` and reset ``_block_matcher``."""
        if value >= 1 and value <= 11 and value % 2:
            self._sad_window_size = value
        else:
            raise InvalidSADWindowSizeError("SADWindowSize must be odd and "
                                            "between 1 and 11.")
        self._replace_bm()

    @property
    def uniquenessRatio(self):
        """Return private ``_uniqueness`` value."""
        return self._uniqueness

    @uniquenessRatio.setter
    def uniquenessRatio(self, value):
        """Set private ``_uniqueness`` and reset ``_block_matcher``."""
        if value >= 5 and value <= 15:
            self._uniqueness = value
        else:
            raise InvalidUniquenessRatioError("Uniqueness ratio must be "
                                              "between 5 and 15.")
        self._replace_bm()

    @property
    def speckleWindowSize(self):
        """Return private ``_speckle_window_size`` value."""
        return self._speckle_window_size

    @speckleWindowSize.setter
    def speckleWindowSize(self, value):
        """Set private ``_speckle_window_size`` and reset ``_block_matcher``."""
        if value >= 0 and value <= 200:
            self._speckle_window_size = value
        else:
            raise InvalidSpeckleWindowSizeError("Speckle window size must be 0 "
                                                "for disabled checks or "
                                                "between 50 and 200.")
        self._replace_bm()

    @property
    def speckleRange(self):
        """Return private ``_speckle_range`` value."""
        return self._speckle_range

    @speckleRange.setter
    def speckleRange(self, value):
        """Set private ``_speckle_range`` and reset ``_block_matcher``."""
        if value >= 0:
            self._speckle_range = value
        else:
            raise InvalidSpeckleRangeError("Speckle range cannot be negative.")
        self._replace_bm()

    @property
    def disp12MaxDiff(self):
        """Return private ``_max_disparity`` value."""
        return self._max_disparity

    @disp12MaxDiff.setter
    def disp12MaxDiff(self, value):
        """Set private ``_max_disparity`` and reset ``_block_matcher``."""
        self._max_disparity = value
        self._replace_bm()

    @property
    def P1(self):
        """Return private ``_P1`` value."""
        return self._P1

    @P1.setter
    def P1(self, value):
        """Set private ``_P1`` and reset ``_block_matcher``."""
        if value < self.P2:
            self._P1 = value
        else:
            raise InvalidFirstDisparityChangePenaltyError("P1 must be less "
                                                          "than P2.")
        self._replace_bm()

    @property
    def P2(self):
        """Return private ``_P2`` value."""
        return self._P2

    @P2.setter
    def P2(self, value):
        """Set private ``_P2`` and reset ``_block_matcher``."""
        if value > self.P1:
            self._P2 = value
        else:
            raise InvalidSecondDisparityChangePenaltyError("P2 must be greater "
                                                          "than P1.")
        self._replace_bm()

    @property
    def fullDP(self):
        """Return private ``_full_dp`` value."""
        return self._full_dp

    @fullDP.setter
    def fullDP(self, value):
        """Set private ``_full_dp`` and reset ``_block_matcher``."""
        self._full_dp = bool(value)
        self._replace_bm()

    def _replace_bm(self):
        """Replace ``_block_matcher`` with current values."""
        self._block_matcher = cv2.StereoSGBM(minDisparity=self._min_disparity,
                        numDisparities=self._num_disp,
                        SADWindowSize=self._sad_window_size,
                        uniquenessRatio=self._uniqueness,
                        speckleWindowSize=self._speckle_window_size,
                        speckleRange=self._speckle_range,
                        disp12MaxDiff=self._max_disparity,
                        P1=self._P1,
                        P2=self._P2,
                        fullDP=self._full_dp)

    def __init__(self, min_disparity=16, num_disp=96, sad_window_size=3,
                 uniqueness=10, speckle_window_size=100, speckle_range=32,
                 p1=216, p2=864, max_disparity=1, full_dp=False,
                 settings=None):
        """Instantiate private variables and call superclass initializer."""
        #: Minimum number of disparities. Normally 0, can be adjusted as
        #: needed
        self._min_disparity = min_disparity
        #: Number of disparities
        self._num_disp = num_disp
        #: Matched block size
        self._sad_window_size = sad_window_size
        #: Uniqueness ratio for found matches
        self._uniqueness = uniqueness
        #: Maximum size of smooth disparity regions to invalid by noise
        self._speckle_window_size = speckle_window_size
        #: Maximum disparity range within connected component
        self._speckle_range = speckle_range
        #: Penalty on disparity change by +-1 between neighbor pixels
        self._P1 = p1
        #: Penalty on disparity change by multiple neighbor pixels
        self._P2 = p2
        #: Maximum left-right disparity. 0 to disable check
        self._max_disparity = max_disparity
        #: Boolean to use full-scale two-pass dynamic algorithm
        self._full_dp = full_dp
        #: StereoSGBM whose state is controlled
        self._block_matcher = cv2.StereoSGBM()
        super(StereoSGBM, self).__init__(settings)

    def get_disparity(self, pair):
        """Compute disparity from image pair (left, right)."""
        return self._block_matcher.compute(pair[0],
                                          pair[1]).astype(np.float32) / 16.0
