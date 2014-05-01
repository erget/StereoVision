#!/bin/env python
'''
Wrapper classes for block matching algorithms.
'''

import cv2
import simplejson

import numpy as np


class BadBlockMatcherArgument(Exception):
    """Bad argument supplied for a ``BlockMatcher``."""
class StereoBMError(BadBlockMatcherArgument):
    """Bad argument supplied for a ``StereoBM``."""
class InvalidBMPreset(StereoBMError):
    """Invalid BM preset."""
class InvalidSearchRange(StereoBMError):
    """Invalid search range."""
class InvalidWindowSize(StereoBMError):
    """Invalid search range."""
class StereoSGBMError(BadBlockMatcherArgument):
    """Bad argument supplied for a ``StereoSGBM``."""
class InvalidNumDisparities(StereoSGBMError):
    """Invalid number of disparities."""
class InvalidSADWindowSize(StereoSGBMError):
    """Invalid search window size."""
class InvalidFirstDisparityChangePenalty(StereoSGBMError):
    """Invalid first disparity change penalty."""
class InvalidSecondDisparityChangePenalty(StereoSGBMError):
    """Invalid second disparity change penalty."""
class InvalidUniquenessRatio(StereoSGBMError):
    """Invalid uniqueness ratio."""
class InvalidSpeckleWindowSize(StereoSGBMError):
    """Invalid speckle window size."""
class InvalidSpeckleRange(StereoSGBMError):
    """Invalid speckle range."""

class BlockMatcher(object):
    """
    Block matching algorithms.

    This abstract exposes the interface for subclasses that wrap OpenCV's block
    matching algorithms. Doing so makes it possible to use them in the strategy
    pattern. In this library, that happens in ``CalibratedPair``, which uses a
    unified interface to interact with any kind of block matcher, and with with
    ``BMTuner``s, which can discover the ``BlockMatcher``'s parameters and
    allow the user to adjust them online.

    Getters and setters are used to protect the block matcher's parameters.
    """
    #: Dictionary of parameter names associated with their maximum values
    parameter_maxima = {}
    def __init__(self, settings=None):
        """Set block matcher parameters and load from file if necessary."""
        #: Block matcher object used for computing point clouds
        self.block_matcher = None
        self.replace_bm()
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
    def replace_bm(self):
        """Replace block matcher with new parameters"""
        raise NotImplementedError
    def compute_disparity(self, image_pair):
        """Compute disparity map from image pair."""
        raise NotImplementedError
    @classmethod
    def compute_3d(cls, disparity, disparity_to_depth_map):
        """Compute point cloud."""
        return cv2.reprojectImageTo3D(disparity, disparity_to_depth_map)

class StereoBM(BlockMatcher):
    """A stereo block matching ``BlockMatcher``."""
    parameter_maxima = {"search_range": None,
                       "window_size": 255,
                       "stereo_bm_preset": cv2.STEREO_BM_NARROW_PRESET}
    @property
    def search_range(self):
        """Number of disparities for ``block_matcher``."""
        return self._search_range
    @search_range.setter
    def search_range(self, value):
        """Set ``search_range`` to multiple of 16, replace ``block_matcher``."""
        if value == 0 or not value % 16:
            self._search_range = value
        else:
            raise InvalidSearchRange("Search range must be a multiple of 16.")
        self.replace_bm()
    @property
    def window_size(self):
        """Search window size."""
        return self._window_size
    @window_size.setter
    def window_size(self, value):
        """Set search window size and update ``block_matcher``."""
        if (value > 4 and
            value < self.parameter_maxima["window_size"] and
            value % 2):
            self._window_size = value
        else:
            raise InvalidWindowSize("Window size must be an odd number between "
                                    "0 and {}.".format(
                                    self.parameter_maxima["window_size"] + 1))
        self.replace_bm()
    @property
    def stereo_bm_preset(self):
        """Stereo BM preset used by ``block_matcher``."""
        return self._bm_preset
    @stereo_bm_preset.setter
    def stereo_bm_preset(self, value):
        """Set stereo BM preset and update ``block_matcher``."""
        if value in (cv2.STEREO_BM_BASIC_PRESET,
                     cv2.STEREO_BM_FISH_EYE_PRESET,
                     cv2.STEREO_BM_NARROW_PRESET):
            self._bm_preset = value
        else:
            raise InvalidBMPreset("Stereo BM preset must be defined as "
                                  "cv2.STEREO_BM_*_PRESET.")
        self.replace_bm()
    def __init__(self, stereo_bm_preset=cv2.STEREO_BM_BASIC_PRESET,
                 search_range=80,
                 window_size=21,
                 settings=None):
        self._bm_preset = cv2.STEREO_BM_BASIC_PRESET
        self._search_range = 0
        self._window_size = 5
        #: OpenCV camera type for ``block_matcher``
        self.stereo_bm_preset = stereo_bm_preset
        #: Number of disparities for ``block_matcher``
        self.search_range = search_range
        #: Search window size for ``block_matcher``
        self.window_size = window_size
        super(StereoBM, self).__init__(settings)
    def replace_bm(self):
        """Replace ``block_matcher`` with current values."""
        self.block_matcher = cv2.StereoBM(preset=self._bm_preset,
                                          ndisparities=self._search_range,
                                          SADWindowSize=self._window_size)
    def compute_disparity(self, pair):
        """
        Compute disparity from image pair (left, right).

        First, convert images to grayscale if needed. Then pass to the
        ``block_matcher`` for stereo matching.
        """
        gray = []
        if pair[0].ndim == 3:
            for side in pair:
                gray.append(cv2.cvtColor(side, cv2.COLOR_BGR2GRAY))
        else:
            gray = pair
        return self.block_matcher.compute(gray[0], gray[1],
                                          disptype=cv2.CV_32F) / 32

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
        return self._min_disparity
    @minDisparity.setter
    def minDisparity(self, value):
        self._min_disparity = value
        self.replace_bm()
    @property
    def numDisparities(self):
        return self._num_disp
    @numDisparities.setter
    def numDisparities(self, value):
        """Set ``numDisparities``, replace block matcher."""
        if value > 0 and value % 16 == 0:
            self._num_disp = value
        else:
            raise InvalidNumDisparities("numDisparities must be a positive "
                                        "integer evenly divisible by 16.")
        self.replace_bm()
    @property
    def SADWindowSize(self):
        return self._sad_window_size
    @SADWindowSize.setter
    def SADWindowSize(self, value):
        """Set search window size, replace block matcher."""
        if value >= 1 and value <= 11 and value % 2:
            self._sad_window_size = value
        else:
            raise InvalidSADWindowSize("SADWindowSize must be odd and between "
                                       "1 and 11.")
        self.replace_bm()
    @property
    def uniquenessRatio(self):
        return self._uniqueness
    @uniquenessRatio.setter
    def uniquenessRatio(self, value):
        if value >= 5 and value <= 15:
            self._uniqueness = value
        else:
            raise InvalidUniquenessRatio("Uniqueness ratio must be between 5 "
                                         "and 15.")
        self.replace_bm()
    @property
    def speckleWindowSize(self):
        return self._speckle_window_size
    @speckleWindowSize.setter
    def speckleWindowSize(self, value):
        if value >= 0 and value <= 200:
            self._speckle_window_size = value
        else:
            raise InvalidSpeckleWindowSize("Speckle window size must be 0 for "
                                           "disabled checks or between 50 and "
                                           "200.")
        self.replace_bm()
    @property
    def speckleRange(self):
        return self._speckle_range
    @speckleRange.setter
    def speckleRange(self, value):
        if value >= 0:
            self._speckle_range = value
        else:
            raise InvalidSpeckleRange("Speckle range cannot be negative.")
        self.replace_bm()
    @property
    def disp12MaxDiff(self):
        return self._max_disparity
    @disp12MaxDiff.setter
    def disp12MaxDiff(self, value):
        self._max_disparity = value
        self.replace_bm()
    @property
    def P1(self):
        return self._P1
    @P1.setter
    def P1(self, value):
        """Set first disparity change penalty, replace block matcher."""
        if value < self.P2:
            self._P1 = value
        else:
            raise InvalidFirstDisparityChangePenalty("P1 must be less than "
                                                     "P2.")
        self.replace_bm()
    @property
    def P2(self):
        return self._P2
    @P2.setter
    def P2(self, value):
        """Set second disparity change penalty, replace block matcher."""
        if value > self.P1:
            self._P2 = value
        else:
            raise InvalidSecondDisparityChangePenalty("P2 must be greater "
                                                      "than P1.")
        self.replace_bm()
    @property
    def fullDP(self):
        return self._full_dp
    @fullDP.setter
    def fullDP(self, value):
        self._full_dp = bool(value)
        self.replace_bm()
    def __init__(self, min_disparity=16, num_disp=96, sad_window_size=3,
                 uniqueness=10, speckle_window_size=100, speckle_range=32,
                 p1=216, p2=864, max_disparity=1, full_dp=False,
                 settings=None):
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
        super(StereoSGBM, self).__init__(settings)
    def replace_bm(self):
        """Replace ``block_matcher`` with current values."""
        self.block_matcher = cv2.StereoSGBM(minDisparity=self._min_disparity,
                        numDisparities=self._num_disp,
                        SADWindowSize=self._sad_window_size,
                        uniquenessRatio=self._uniqueness,
                        speckleWindowSize=self._speckle_window_size,
                        speckleRange=self._speckle_range,
                        disp12MaxDiff=self._max_disparity,
                        P1=self._P1,
                        P2=self._P2,
                        fullDP=self._full_dp)
    def compute_disparity(self, pair):
        """Compute disparity from image pair (left, right)."""
        return self.block_matcher.compute(pair[0],
                                          pair[1]).astype(np.float32) / 16.0
