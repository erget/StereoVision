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

from setuptools import setup

setup(name="StereoVision",
      version="1.0.3",
      description=("Library and utilities for 3d reconstruction from stereo "
                   "cameras."),
      long_description=open("README.rst").read(),
      author="Daniel Lee",
      author_email="lee.daniel.1986@gmail.com",
      packages=["stereovision"],
      scripts=["bin/calibrate_cameras",
               "bin/capture_chessboards",
               "bin/images_to_pointcloud",
               "bin/show_webcams",
               "bin/tune_blockmatcher"],
      url="http://erget.github.com/StereoVision",
      download_url="http://pypi.python.org/pypi/StereoVision",
      license="GNU GPL",
      requires=["cv2",
                "simplejson",
                "numpy",
                "progressbar"],
      provides=["stereovision"],
      classifiers=["Development Status :: 5 - Production/Stable",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 2",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Education",
                   "Intended Audience :: Science/Research",
                   "License :: Freely Distributable",
                   "License :: OSI Approved :: GNU General Public License v3 "
                                              "or later (GPLv3+)",
                   "Natural Language :: English",
                   "Operating System :: OS Independent",
                   "Programming Language :: Python :: 2.7",
                   "Topic :: Multimedia :: Graphics :: Capture"])
