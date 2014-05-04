Using StereoVision
==================

``StereoVision`` offers a number of command line utilities that you can use in
order to produce 3d point clouds from stereo images. They are listed in the
following table, roughly in the order that you would use them if you were to
set up a project of your own from scratch. Usage information can be obtained on
the command line by calling them with the ``-h`` and ``--help`` flags.

    ========================    ===============================================
    Script name                 Purpose
    ========================    ===============================================
    ``show_webcams``            Show output from stereo camera pair, optionally
                                capture images
    ``capture_chessboards``     Capture images of chessboards simultaneously
                                visible from both cameras in stereo pair for
                                the purpose of calibrating the camera pair.
                                Optionally calibrate camera pair online.
    ``calibrate_cameras``       Calibrate stereo pair using previously captured
                                chessboard images
    ``tune_blockmatcher``       Manually tune block matching algorithm to
                                produce good disparity maps with a given stereo
                                pair
    ``images_to_pointcloud``    Convert image pairs captured with a calibrated
                                stereo pair to a colored 3d point cloud
    ========================    ===============================================

These scripts can also be used as examples for how to use the classes in
``StereoVision``.

 If you'd like to use the library classes available in ``StereoVision``, see
 the `developer documentation <development.html>`_.
