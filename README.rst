StereoVision: Library and utilities for 3d reconstruction from stereo cameras
=============================================================================

StereoVision is a package for working with stereo cameras, especially with the
intent of using them to produce 3D point clouds. The focus is on performance,
ease of usability, and the ability to construct 3D imaging setups cheaply.

StereoVision relies heavily on OpenCV. If you're not sure about what a given
variable does or what values would make sense for it and no explanation is
provided in the StereoVision documentation, refer to OpenCV's documentation in
order to better understand how they work.

It's available on PyPI, so you can install it like this::

    pip install StereoVision

Tutorials are available on the Stackable blog:

- `Building a stereo rig`_
- `Stereo calibration`_
- `Tuning the block matcher`_
- `Producing point clouds`_

If you find a bug or would like to request a feature, please `report it with
the issue tracker <https://github.com/erget/StereoVision/issues>`_. If you'd
like to contribute to StereoVision, feel free to `fork it on GitHub
<https://github.com/erget/StereoVision>`_.

StereoVision is released under the GNU General Public License, so feel free to
use it any way you like. It would be nice to let me know if you do anything
cool with it though.

Author: `Daniel Lee <Lee.Daniel.1986@gmail.com>`_

.. _Building a stereo rig: https://erget.wordpress.com/2014/02/01/calibrating-a-stereo-camera-with-opencv/
.. _Stereo calibration: https://erget.wordpress.com/2014/02/28/calibrating-a-stereo-pair-with-python/
.. _Tuning the block matcher: https://erget.wordpress.com/2014/05/02/producing-3d-point-clouds-from-stereo-photos-tuning-the-block-matcher-for-best-results/
.. _Producing point clouds: https://erget.wordpress.com/2014/04/27/producing-3d-point-clouds-with-a-stereo-camera-in-opencv
