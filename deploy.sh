#!/bin/bash
cd ../StereoVision/doc/
make html
cd -
cp -r ../StereoVision/doc/_build/html/* .
git add .
git commit -am "Update documentation"
git push
