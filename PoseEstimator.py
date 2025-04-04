import pyrealsense2 as rs
import numpy as np
import cv2

"""
ID: Desired Rotation
37: -90 around x axis
3: -90 around x axis, then -90 around z
1: 180 around y axis, then -90 around x
17: 90 around y, then -90 around x
"""

# Helper rotation vectors (in radians)
rxN90 = np.array([-np.pi/2, 0, 0])     # -90° around X
ry180 = np.array([0, np.pi, 0])        # 180° around Y
ry90  = np.array([0, np.pi/2, 0])      # 90° around Y
rzN90 = np.array([0, 0, -np.pi/2])    # -90° around Z

id_rotation_adjustments = {
    37: cv2.Rodrigues(rxN90)[0],
    3: cv2.Rodrigues(rxN90)[0] @ cv2.Rodrigues(rzN90)[0],
    1: cv2.Rodrigues(ry180)[0] @ cv2.Rodrigues(rxN90)[0],
    17: cv2.Rodrigues(ry90)[0] @ cv2.Rodrigues(rxN90)[0],
}

M2IN = 39.37 # meters to inches

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

# camera_matrix =  [[604.44916   0.      330.29477]
#  [  0.      603.9991  245.70053]
#  [  0.        0.        1.     ]]

# Define ArUco marker properties
marker_size = 0.054  # Marker size in meters (adjust to your actual marker size)

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
    print(f"Detected IDs: {ids}")


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

            # Convert detected rvec to a rotation matrix
            Rotation_marker, _ = cv2.Rodrigues(rvec)

            # Get the predefined adjustment for this marker ID
            Rotation_adjusted = id_rotation_adjustments.get(ids[i][0], np.eye(3))  # Default to identity if not in dict

            # Compute the new rvec: combine detected rotation with the desired adjustment
            Rotation_final = Rotation_marker @ Rotation_adjusted  # Apply the predefined correction
            rvec_final, _ = cv2.Rodrigues(Rotation_final)  # Convert back to rotation vector

            # Define the center offset (translation in local marker frame)
            if ids[i][0] == 2:
                center_box_offset = np.array([[0, 0, 0.075]], dtype=np.float32)  # Move up in marker's Z
            else:
                center_box_offset = np.array([[0, 0, -0.075]], dtype=np.float32)  # Move down in marker's Z

            # Transform the center offset into the camera frame
            center_box_tvec = tvec + Rotation_marker @ center_box_offset.T  # Apply translation in marker's frame

            # Draw the coordinate system at the center of the box
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec_final, center_box_tvec, marker_size / 2)

            # Draws Location of Center Box
            center_box_2D, _ = cv2.projectPoints(center_box_offset[0], rvec, tvec, camera_matrix, dist_coeffs)
            box_center_x, box_center_y = int(center_box_2D[0][0][0]), int(center_box_2D[0][0][1])
            cv2.circle(color_image, (box_center_x, box_center_y), 5, (0, 255, 0), -1)  # Draw green circles at the corners

            # if success:

            # Draw marker and axis
            cv2.aruco.drawDetectedMarkers(color_image, [corners[i]], ids[i])
            cv2.drawFrameAxes(color_image, camera_matrix, dist_coeffs, rvec, tvec, marker_size / 2)

            # Get depth value
            center_x = int(np.mean(marker_corners[:, 0]))  # X-coordinate of the marker center
            center_y = int(np.mean(marker_corners[:, 1]))  # Y-coordinate of the marker center
            depth_m = depth_frame.get_distance(center_x, center_y)

            # Display depth on the image
            if (depth_m == 0 or depth_m > 1):
                    continue
            cv2.putText(color_image, f"Depth: {depth_m*M2IN:.2f} in", 
                        (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Show the video feed with detected ArUco markers and pose
    cv2.imshow("RealSense ArUco Detection with Depth", color_image)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close windows
pipeline.stop()
cv2.destroyAllWindows()
