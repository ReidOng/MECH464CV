import pyrealsense2 as rs
import numpy as np
import cv2

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
                # Extract one corner (e.g., top-left corner)
                x, y = int(marker_corners[0][0][0]), int(marker_corners[0][0][1])
                
                # Get depth at that pixel (convert from mm to meters)
                depth = depth_frame.get_distance(x, y)

                # Draw marker and depth text
                cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
                cv2.putText(color_image, f"Depth: {depth:.2f}m", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Marker {ids[i]} - Corner (x={x}, y={y}) -> Depth: {depth:.2f}m")

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
