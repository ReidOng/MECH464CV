'''
Welcome to the ArUco Marker Generator!
  
This program:
  - Generates ArUco markers using OpenCV and Python
'''
  
from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
  
# Project: ArUco Marker Generator
# Date created: 12/17/2021
# Python version: 3.8
# Reference: https://www.pyimagesearch.com/2020/12/14/generating-aruco-markers-with-opencv-and-python/
 
marker_id = 17
output_filename = "aruco_id1.png"
  
def main():
  """
  Main method of the program.
  """
  dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
  marker_size = 200
  marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)


  # Save the ArUco tag to the current directory
  cv2.imwrite(output_filename, marker_image)
  cv2.imshow("ArUco Marker", marker_image)
  cv2.waitKey(0)
   
if __name__ == '__main__':
  print(__doc__)
  main()