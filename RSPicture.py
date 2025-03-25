import pyrealsense2 as rs
import numpy as np
import cv2
import os

"""
Use this script to take photos of the checkerboard for the realsense calibration
"""

# Set the folder where images will be saved
SAVE_FOLDER = "C:/Users/reido/MECH 464/Calibration"  # Change this path

# Ensure the save directory exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
count = 1 # image counter

print("Press 's' to save an image, 'q' to quit.")

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply color map to depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack images side by side
        images = np.hstack((color_image, depth_colormap))

        # Show the camera feed
        cv2.imshow("RealSense Camera", images)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF

        # Save image when 's' is pressed
        if key == ord('s'):
            image_filename = os.path.join(SAVE_FOLDER, "realsense_capture_" + str(count) + ".png")
            count += 1
            cv2.imwrite(image_filename, color_image)
            print(f"Image saved: {image_filename}")

        # Exit when 'q' is pressed
        elif key == ord('q'):
            print("Closing camera...")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
