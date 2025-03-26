import pyrealsense2 as rs
import numpy as np
import cv2

'''
This script uses the RealSense's RGB and Depth Cameras to detect an Aruco Marker, and livestream its position onto the camera view.
'''

M2IN = 39.37

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Configure depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Define ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Get Intrinsics
profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

try:
    while True:
        # Wait for frames and align depth to color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale for ArUco detection
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        # print(ids)
        # print(corners)

        if ids is not None:
            for i in range(len(ids)): # loop through all detected markers
                marker_corners = corners[i]  # Get four corner points of the current ID
                # print(marker_corners)
                
                # Get the location of the top left corner
                corner = 0
                x, y = int(marker_corners[0][corner][0]), int(marker_corners[0][corner][1])
                
                # Get the depth and calculate position of that pixel (convert from mm to meters)
                depth = depth_frame.get_distance(x, y)
                point_3D = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                X, Y, Z = point_3D # X points to the right of the sensor, y points down, Z towards image frame

                # Draw marker and depth text
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.putText(color_image, f"Depth: {depth*M2IN:.2f}in", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Marker {ids[i]} - Corner (x={x}, y={y}) -> Depth: {depth*M2IN:.2f}\" at [{X*M2IN:.2f} {Y*M2IN:.2f} {Z*M2IN:.2f}]")

        # Show images
        # cv2.imshow("Depth Frame", depth_image)
        cv2.imshow("Color Frame", color_image)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
