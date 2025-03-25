import cv2
import numpy as np
from matplotlib import pyplot as plt
import requests
from io import BytesIO
from PIL import Image

''' FOR CAMERA CAPTURES
# Connect to capture device
cap = cv2.VideoCapture(1)
# Get a frame from the capture device
# NOTE: ret will be false if the capture is unsuccessful
#       use cap.release()
ret, image = cap.read()

# Check if frame was captured
if not ret:
    print("Error: Could not capture image from camera.")
    exit()

# show the frame
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
cap.release()
'''

# # Load the image
image = cv2.imread('C:\\Users\\reido\\MECH 464\\Captureofmarker_42.png')
# # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# # plt.show()

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
# parameters = cv2.aruco.DetectorParameters()

# # Create the ArUco detector
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
# # Detect the markers
# corners, ids, rejected = detector.detectMarkers(gray)
# # Print the detected markers
# print("Detected markers:", ids)

# if ids is not None:
#     cv2.aruco.drawDetectedMarkers(image, corners, ids)
#     cv2.imshow('Detected Markers', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Check the number of channels in the image
if len(image.shape) == 2:  # Grayscale image
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
elif image.shape[2] == 4:  # RGBA image
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Create the ArUco detector
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Detect the markers
corners, ids, rejected = detector.detectMarkers(gray)
print("Detected markers:", ids)

if ids is not None:
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    cv2.imshow('Detected Markers', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()