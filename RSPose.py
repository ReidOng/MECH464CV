import pyrealsense2 as rs
import numpy as np
import cv2

'''
This script uses the RealSense's RGB and Depth Cameras to detect an Aruco Marker, and livestream its position onto the camera view.
'''

def compute_rvec_from_points(origin, x_point, y_point):
    """
    Compute the rotation vector (rvec) defining a coordinate frame given three points in space.
    
    Parameters:
    - origin: (3,) array, the origin of the coordinate frame
    - x_point: (3,) array, a point along the x-axis
    - y_point: (3,) array, a point along the y-axis
    
    Returns:
    - rvec: (3,1) array, the rotation vector representing the orientation
    """
    # Compute unit x-axis
    x_axis = x_point - origin
    x_axis /= np.linalg.norm(x_axis)

    # Compute preliminary y-axis
    y_axis = y_point - origin
    y_axis /= np.linalg.norm(y_axis)

    # Compute z-axis as cross product of x and y
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Recompute y-axis to ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)

    # Construct rotation matrix
    R = np.column_stack((x_axis, y_axis, z_axis))

    # Convert rotation matrix to rotation vector
    rvec, _ = cv2.Rodrigues(R)

    return rvec

if(__name__== "__main__"):

    M2IN = 39.37 # Convert meter to inches

    # Initialize Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Configure depth and color streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get Camera Intrinsics
    color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    camera_matrix = np.array([
        [color_intrinsics.fx, 0, color_intrinsics.ppx],  # Focal length X, Principal point X
        [0, color_intrinsics.fy, color_intrinsics.ppy],  # Focal length Y, Principal point Y
        [0, 0, 1]                            # Homogeneous coordinate
    ])
    dist_coeffs = np.array(color_intrinsics.coeffs)  # Distortion parameters

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
            align = rs.align(rs.stream.color)
            frames = align.process(pipeline.wait_for_frames())
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
                    
                    # Get the location of the top left corner (origin)
                    corner = 0
                    x, y = int(marker_corners[0][0][0]), int(marker_corners[0][0][1])
                    depth = depth_frame.get_distance(x, y)
                    
                    # Filter out incorrect readings
                    if (depth == 0 or depth > 3):
                        continue
                    
                    origin = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)) # in units [m]
                    X, Y, Z = origin # X points to the right of the sensor, y points down, Z towards image frame
                    
                    # Get the location of the top right corner (y-axis)
                    x_y, y_y = int(marker_corners[0][1][0]), int(marker_corners[0][1][1])
                    y_point = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_y, y_y], depth_frame.get_distance(x_y, y_y)))
                    
                    # Get the location of the bottom left corner (x-axis)
                    x_x, y_x = int(marker_corners[0][3][0]), int(marker_corners[0][3][1])
                    x_point = np.array(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_x, y_x], depth_frame.get_distance(x_x, y_x)))

                    # Compute the rotation vector
                    rvec = compute_rvec_from_points(origin, x_point, y_point)

                    # Draw the Coordinate frame on the marker
                    cv2.drawFrameAxes(color_image, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs, rvec=rvec, tvec=origin, length=0.025) 

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
