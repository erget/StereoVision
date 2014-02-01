#!/bin/env python

import argparse

import cv2

class StereoPair(object):
    """
    A stereo pair of cameras.
    
    Should be initialized with a context manager to ensure that the cameras are
    freed properly after use.
    """
    
    def __init__(self, devices):
        """
        Initialize cameras.
        
        ``devices`` is an iterable containing the device numbers.
        """
        self.captures = [cv2.VideoCapture(device) for device in devices]
        self.windows = ["Window {}".format(device) for device in devices]
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        for capture in self.captures:
            capture.release()
    
    def get_frames(self):
        """Get current frames from cameras."""
        return [capture.read()[1] for capture in self.captures]
    
    def show_frames(self, wait=0):
        """
        Show current frames from cameras.
        
        ``wait`` is the wait interval before the window closes.
        """
        for window, frame in zip(self.windows, self.get_frames()):
            cv2.imshow(window, frame)
        cv2.waitKey(wait)
        
    def show_videos(self):
        """Show video from cameras."""
        while True:
            self.show_frames(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def main():
    """
    Show the video from two webcams successively.

    For best results, connect the webcams while starting the computer.
    I have noticed that in some cases, if the webcam is not already connected
    when the computer starts, the USB device runs out of memory. Switching the
    camera to another USB port has also caused this problem in my experience.
    """
    parser = argparse.ArgumentParser(description="Show video from two "
                                     "webcams.\n\nPress 'q' to exit.")
    parser.add_argument("devices", type=int, nargs=2, help="Device numbers "
                        "for the cameras that should be accessed.")
    args = parser.parse_args()

    with StereoPair(args.devices) as pair:
        pair.show_videos()
    print("Cameras closed.")

if __name__ == "__main__":
    main()
