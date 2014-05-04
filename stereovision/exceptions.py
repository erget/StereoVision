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
Various exceptions for working with stereovision.

Classes:

    * ``ChessboardNotFoundError``

    * ``BadBlockMatcherArgumentError``

        * ``StereoBMError``

            * ``InvalidBMPresetError``
            * ``InvalidSearchRangeError``
            * ``InvalidWindowSizeError``

        * ``StereoSGBMError``

            * ``InvalidNumDisparitiesError``
            * ``InvalidSADWindowSizeError``
            * ``InvalidFirstDisparityChangePenaltyError``
            * ``InvalidSecondDisparityChangePenaltyError``
            * ``InvalidUniquenessRatioError``
            * ``InvalidSpeckleWindowSizeError``
            * ``InvalidSpeckleRangeError``

.. image:: classes_exceptions.svg
    :width: 100%
"""

class ChessboardNotFoundError(Exception):
    """No chessboard could be found in searched image."""


class BadBlockMatcherArgumentError(Exception):
    """Bad argument supplied for a ``BlockMatcher``."""

class StereoBMError(BadBlockMatcherArgumentError):
    """Bad argument supplied for a ``StereoBM``."""

class StereoSGBMError(BadBlockMatcherArgumentError):
    """Bad argument supplied for a ``StereoSGBM``."""

class InvalidBMPresetError(StereoBMError):
    """Invalid BM preset."""

class InvalidSearchRangeError(StereoBMError):
    """Invalid search range."""

class InvalidWindowSizeError(StereoBMError):
    """Invalid search range."""

class InvalidNumDisparitiesError(StereoSGBMError):
    """Invalid number of disparities."""

class InvalidSADWindowSizeError(StereoSGBMError):
    """Invalid search window size."""

class InvalidFirstDisparityChangePenaltyError(StereoSGBMError):
    """Invalid first disparity change penalty."""

class InvalidSecondDisparityChangePenaltyError(StereoSGBMError):
    """Invalid second disparity change penalty."""

class InvalidUniquenessRatioError(StereoSGBMError):
    """Invalid uniqueness ratio."""

class InvalidSpeckleWindowSizeError(StereoSGBMError):
    """Invalid speckle window size."""

class InvalidSpeckleRangeError(StereoSGBMError):
    """Invalid speckle range."""
