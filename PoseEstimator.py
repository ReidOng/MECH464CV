import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Get camera intrinsics for depth stream
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],  
                          [0, intrinsics.fy, intrinsics.ppy],  
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5,))  # Assume no distortion for RealSense

# Define ArUco marker properties
marker_size = 0.055  # Marker size in meters (adjust to your actual marker size)

# Define ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

while True:
    # Wait for frames from the RealSense camera
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue  # Skip iteration if frames are not ready

    # Convert to NumPy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # Convert to grayscale for ArUco detection
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, _ = aruco_detector.detectMarkers(gray)


    if ids is not None:
        for i in range(len(ids)):
            # Extract marker corner points
            marker_corners = corners[i].reshape(4, 2)

            # Define 3D object points of the marker in local coordinates
            obj_points = np.array([
                [-marker_size / 2,  marker_size / 2, 0],  # Top-left
                [ marker_size / 2,  marker_size / 2, 0],  # Top-right
                [ marker_size / 2, -marker_size / 2, 0],  # Bottom-right
                [-marker_size / 2, -marker_size / 2, 0]   # Bottom-left
            ], dtype=np.float32)

            # SolvePnP to get rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners, camera_matrix, dist_coeffs)

            if success:
                # Draw marker and axis
                cv2.aruco.drawDetectedMarkers(color_image, [corners[i]], ids[i])
                cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)

                # Get marker position in meters
                print(f"Marker ID {ids[i][0]} Pose:\nRotation:\n{rvec}\nTranslation:\n{tvec}")

    # Show the video feed
    cv2.imshow("RealSense ArUco Detection", color_image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close windows
pipeline.stop()
cv2.destroyAllWindows()
