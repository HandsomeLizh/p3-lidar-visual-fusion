# Incrementally updated 2.5D semantic map.
# All arrays use row-major order: index = row * width + column.

std_msgs/Header header

uint32 update_id
float32 resolution
uint32 width
uint32 height

# Pose of the lower-left grid corner in header.frame_id.
geometry_msgs/Pose origin

# Standard occupancy convention: -1 unknown, 0 free, 100 occupied.
int8[] occupancy

# Semantic class ID and confidence [0, 1].
uint8[] semantic
float32[] semantic_confidence

# 2.5D layers. Invalid/unobserved values are NaN.
float32[] elevation
float32[] elevation_variance
float32[] height_range
float32[] roughness

# Number of accepted 3D observations fused into each cell.
uint32[] observation_count

string[] class_names
