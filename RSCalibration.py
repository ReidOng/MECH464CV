import pyrealsense2 as rs

# Start RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)

# Get depth stream profile and extract intrinsics
profile = pipeline.get_active_profile()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Print intrinsics
print("Camera Intrinsics:")
print(f"Width: {depth_intrinsics.width}, Height: {depth_intrinsics.height}")
print(f"Focal Length (fx, fy): ({depth_intrinsics.fx}, {depth_intrinsics.fy})")
print(f"Principal Point (cx, cy): ({depth_intrinsics.ppx}, {depth_intrinsics.ppy})")
print(f"Distortion Model: {depth_intrinsics.model}")
print(f"Distortion Coefficients: {depth_intrinsics.coeffs}")

# Stop pipeline
pipeline.stop()
