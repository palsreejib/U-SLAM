#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped, Point, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from visualization_msgs.msg import MarkerArray, Marker
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, StaticTransformBroadcaster
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from collections import deque
import os
import struct
import math
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class Particle:
    def __init__(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        self.position = np.array([x, y, z])
        self.yaw = yaw
        self.weight = 1.0
        
    def as_matrix(self):
        """Convert particle to 4x4 transformation matrix"""
        mat = np.eye(4)
        mat[:3, 3] = self.position
        mat[:3, :3] = R.from_euler('z', self.yaw).as_matrix()
        return mat

class UnderwaterSLAM(Node):
    def __init__(self):
        super().__init__('underwater_slam')
        
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        # Initialize components
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Configuration
        self.declare_parameters(
            namespace='',
            parameters=[
                ('map_resolution', 0.1),
                ('map_size', 100),
                ('num_particles', 500),
                ('enable_visual_slam', True),
                ('enable_sonar_mapping', True)
            ]
        )
        
        # Get parameters
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_size = self.get_parameter('map_size').value
        self.num_particles = self.get_parameter('num_particles').value
        self.enable_visual_slam = self.get_parameter('enable_visual_slam').value
        self.enable_sonar_mapping = self.get_parameter('enable_sonar_mapping').value
        
        # Frame names
        self.map_frame = "map"
        self.odom_frame = "odom"
        self.base_frame = "base_link"
        self.camera_frame = "camera_optical_frame"
        self.dvl_frame = "dvl_frame"
        self.sonar_frame = "sonar_frame"
        
        # State variables
        self.current_pose = np.eye(4)
        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=np.int8)
        self.point_cloud = None
        self.last_image = None
        self.last_dvl_velocity = np.zeros(3)
        self.particles = []
        self.initialize_particles()
        
        # Visual SLAM state
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.map_points = []
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        self.dvl_sub = self.create_subscription(
            TwistStamped, 'dvl/velocity', self.dvl_callback, 10)
        self.sonar_sub = self.create_subscription(
            PointCloud2, 'sonar/pointcloud', self.sonar_callback, 10)
            
        # Publishers
        self.map_pub = self.create_publisher(OccupancyGrid, 'slam/map', 10)
        self.odom_pub = self.create_publisher(Odometry, 'slam/odometry', 10)
        self.pose_pub = self.create_publisher(PoseStamped, 'slam/pose', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'slam/map_markers', 10)

        self.publish_static_map_transform()
        
        # Timer for periodic updates
        self.timer = self.create_timer(0.1, self.update_slam)
        
        self.get_logger().info("Underwater SLAM node initialized")

        # Initialize TF buffer and listener (add this)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Map initialization (replace existing)
        self.occupancy_grid = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  # -1 = unknown
        
        # Add these parameters (new)
        self.declare_parameter('occupancy_threshold', 50)  # Cells >50 = occupied
        self.declare_parameter('free_threshold', 20)       # Cells <20 = free
        self.occupancy_thresh = self.get_parameter('occupancy_threshold').value
        self.free_thresh = self.get_parameter('free_threshold').value
        
        # Sensor status flags (new)
        self.dvl_initialized = False
        self.sonar_initialized = False
        self.camera_initialized = False

    def initialize_particles(self):
        """Initialize particles with uniform distribution"""
        self.particles = []
        for _ in range(self.num_particles):
            x = np.random.uniform(-5.0, 5.0)
            y = np.random.uniform(-5.0, 5.0)
            z = 0.0  # Assuming mostly 2D motion
            yaw = np.random.uniform(-np.pi, np.pi)
            self.particles.append(Particle(x, y, z, yaw))
            
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.last_image = gray_image  # Store grayscale for processing
            
            # Get camera transform (NEW)
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                msg.header.stamp)
            
            self.latest_camera_tf = transform
            self.camera_initialized = True
            
            # Feature extraction (existing ORB code)
            keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
            if self.prev_keypoints and self.prev_descriptors:
                matches = self.bf.match(self.prev_descriptors, descriptors)
                if len(matches) > 10:
                    # Estimate motion from feature matches
                    src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in matches])
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches])
                    
                    # Essential matrix estimation
                    E, mask = cv2.findEssentialMat(
                        src_pts, dst_pts, 
                        focal=1.0, pp=(0.0, 0.0),
                        method=cv2.RANSAC, prob=0.999, threshold=1.0)
                        
                    if E is not None:
                        # Recover pose
                        _, R, t, mask = cv2.recoverPose(
                            E, src_pts, dst_pts)
                            
                        # Convert to transformation matrix
                        transform = np.eye(4)
                        transform[:3, :3] = R
                        transform[:3, 3] = t.flatten()
                        
                        # Update particles with visual odometry
                        self.apply_visual_update(transform)
                        
            # Store current features for next frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
        except (TransformException, cv2.error) as e:
            self.get_logger().warn(f"Image processing error: {str(e)}")

    def update_with_visual_features(self):
        """Use visual features to improve SLAM"""
        if not (self.camera_initialized and self.prev_keypoints):
            return
        
        try:
            # Get camera pose in map frame
            transform = self.latest_camera_tf
            camera_pos = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ])
            
            # Update particles based on visual features (simplified)
            for particle in self.particles:
                # Compare expected vs actual features
                # (This is simplified - use your actual visual odometry here)
                particle.weight *= self.calculate_feature_match_confidence(particle)
                
        except Exception as e:
            self.get_logger().warn(f"Visual update failed: {str(e)}")

    def apply_visual_update(self, transform):
        """Update particles based on visual odometry"""
        for particle in self.particles:
            # Convert particle to matrix
            particle_mat = particle.as_matrix()
            
            # Apply transform
            updated_mat = particle_mat @ transform
            
            # Extract new position and orientation
            particle.position = updated_mat[:3, 3]
            particle.yaw = R.from_matrix(updated_mat[:3, :3]).as_euler('zyx')[0]
            
    def dvl_callback(self, msg):
        """Process DVL velocity measurements"""
        try:
            self.last_dvl_velocity = np.array([
                msg.twist.linear.x,
                msg.twist.linear.y,
                msg.twist.angular.z
            ])
            
            # Predict particle motion based on DVL
            self.predict_particles()
            
        except Exception as e:
            self.get_logger().error(f"DVL processing error: {str(e)}")

    def predict_particles(self):
        """Predict particle motion based on DVL velocity"""
        dt = 0.1  # Time step
        
        for particle in self.particles:
            # Add noise to velocity
            noisy_vx = self.last_dvl_velocity[0] + np.random.normal(0, 0.05)
            noisy_vy = self.last_dvl_velocity[1] + np.random.normal(0, 0.05)
            noisy_yaw = self.last_dvl_velocity[2] + np.random.normal(0, 0.01)
            
            # Update position (in world frame)
            particle.position[0] += (noisy_vx * np.cos(particle.yaw) - 
                                   noisy_vy * np.sin(particle.yaw)) * dt
            particle.position[1] += (noisy_vx * np.sin(particle.yaw) + 
                                   noisy_vy * np.cos(particle.yaw)) * dt
            particle.yaw += noisy_yaw * dt
            
            # Add some random noise
            particle.position += np.random.normal(0, 0.01, size=3)
            particle.yaw += np.random.normal(0, 0.005)

    def sonar_callback(self, msg):
        """Process sonar point cloud data"""
        try:
            points = self.pointcloud2_to_array(msg)
            self.point_cloud = points
            
            if not self.enable_sonar_mapping:
                return
                
            # Update particle weights based on sonar measurements
            self.update_particle_weights(points)
            
            # Update occupancy grid
            self.update_occupancy_grid(points)
            
            # Resample particles if needed
            if self.effective_particles() < self.num_particles / 2:
                self.resample_particles()
                
        except Exception as e:
            self.get_logger().error(f"Sonar processing error: {str(e)}")

    def update_particle_weights(self, points):
        """Update particle weights based on sonar measurements"""
        if points is None or len(points) == 0:
            return
            
        # Simple observation model - prefer particles where points are consistent
        for particle in self.particles:
            # Transform points to particle frame
            transformed = self.transform_points(points, particle)
            
            # Calculate weight based on point distribution
            # (This is simplified - real implementation would use a proper sensor model)
            dist = np.linalg.norm(transformed, axis=1)
            particle.weight = 1.0 / (np.mean(dist) + 1e-6)
            
        # Normalize weights
        total_weight = sum(p.weight for p in self.particles)
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight

    def resample_particles(self):
        """Resample particles based on their weights"""
        weights = np.array([p.weight for p in self.particles])
        indices = np.random.choice(
            range(self.num_particles),
            size=self.num_particles,
            p=weights,
            replace=True)
            
        new_particles = []
        for i in indices:
            # Create new particle with some noise
            p = self.particles[i]
            new_p = Particle(
                p.position[0] + np.random.normal(0, 0.02),
                p.position[1] + np.random.normal(0, 0.02),
                p.position[2],
                p.yaw + np.random.normal(0, 0.01))
            new_p.weight = 1.0 / self.num_particles
            new_particles.append(new_p)
            
        self.particles = new_particles

    def effective_particles(self):
        """Calculate effective number of particles"""
        weights = np.array([p.weight for p in self.particles])
        return 1.0 / np.sum(weights**2)

    def update_occupancy_grid(self, points):
        """Continuously update map without resetting"""
        if points is None or len(points) == 0:
            return
            
        best_particle = max(self.particles, key=lambda p: p.weight)
        robot_pos = best_particle.position[:2]
        
        grid_center = np.array([self.map_size//2, self.map_size//2])
        robot_grid = (robot_pos / self.map_resolution + grid_center).astype(int)
        
        for point in points:
            point_grid = (point[:2] / self.map_resolution + grid_center).astype(int)
            
            # Update observed cells (0=free, 100=occupied)
            if 0 <= point_grid[0] < self.map_size and 0 <= point_grid[1] < self.map_size:
                # Mark as occupied with growing confidence
                self.occupancy_grid[point_grid[0], point_grid[1]] = min(
                    100, self.occupancy_grid[point_grid[0], point_grid[1]] + 20)
                    
            # Mark free space along the ray
            self.bresenham_line(robot_grid, point_grid)

    def bresenham_line(self, start, end):
        """Bresenham's line algorithm for marking free space"""
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.occupancy_grid[x, y] = max(
                        -1, self.occupancy_grid[x, y] - 5)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.occupancy_grid[x, y] = max(
                        -1, self.occupancy_grid[x, y] - 5)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

    def update_slam(self):
        """Main SLAM update loop"""
        try:
            # Update best pose estimate
            best_particle = max(self.particles, key=lambda p: p.weight)
            self.current_pose = best_particle.as_matrix()
            
            # Publish transforms
            self.publish_transforms()
            
            # Publish occupancy grid
            self.publish_occupancy_grid()
            
            # Publish odometry and pose
            self.publish_odometry()
            
            # Publish visualization markers
            self.publish_markers()
            
        except Exception as e:
            self.get_logger().error(f"SLAM update error: {str(e)}")

    def publish_transforms(self):
        """Publish TF transforms"""
        # Map to odom transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        
        # Set transform from current pose
        transform.transform.translation.x = self.current_pose[0, 3]
        transform.transform.translation.y = self.current_pose[1, 3]
        transform.transform.translation.z = self.current_pose[2, 3]
        
        rot = R.from_matrix(self.current_pose[:3, :3])
        quat = rot.as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(transform)

    def publish_occupancy_grid(self):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.map_frame
        grid_msg.info.resolution = self.map_resolution
        grid_msg.info.width = self.map_size
        grid_msg.info.height = self.map_size
        
        # Center the map origin
        grid_msg.info.origin.position.x = -self.map_size * self.map_resolution / 2
        grid_msg.info.origin.position.y = -self.map_size * self.map_resolution / 2
        grid_msg.info.origin.orientation.w = 1.0
        
        # Convert to proper ROS format (-1=unknown, 0=free, 100=occupied)
        grid_data = (self.occupancy_grid.T).astype(np.int8).flatten().tolist()
        grid_msg.data = grid_data
        
        self.map_pub.publish(grid_msg)

    def publish_static_map_transform(self):
        """Publish initial static transform from map to odom"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.odom_frame
        transform.transform.rotation.w = 1.0  # Identity rotation
        self.static_tf_broadcaster.sendTransform(transform)

    def publish_odometry(self):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        
        # Set pose
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]
        
        rot = R.from_matrix(self.current_pose[:3, :3])
        quat = rot.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        
        # Set velocity (from DVL)
        odom_msg.twist.twist.linear.x = self.last_dvl_velocity[0]
        odom_msg.twist.twist.linear.y = self.last_dvl_velocity[1]
        odom_msg.twist.twist.angular.z = self.last_dvl_velocity[2]
        
        self.odom_pub.publish(odom_msg)
        
        # Also publish as PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = odom_msg.header
        pose_msg.pose = odom_msg.pose.pose
        self.pose_pub.publish(pose_msg)

    def publish_markers(self):
        """Publish visualization markers for particles and features"""
        # Particle markers
        marker_array = MarkerArray()
        
        # Particles
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "particles"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        for particle in self.particles:
            p = Point()
            p.x = particle.position[0]
            p.y = particle.position[1]
            p.z = particle.position[2]
            marker.points.append(p)
            
        marker_array.markers.append(marker)
        
        # Best particle
        best_particle = max(self.particles, key=lambda p: p.weight)
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_particle"
        marker.id = 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.5
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        marker.pose.position.x = best_particle.position[0]
        marker.pose.position.y = best_particle.position[1]
        marker.pose.position.z = best_particle.position[2]
        
        rot = R.from_euler('z', best_particle.yaw)
        quat = rot.as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        marker_array.markers.append(marker)
        
        self.markers_pub.publish(marker_array)

    def pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array with robust parsing"""
        try:
            # Calculate expected point size based on fields
            point_size = 0
            field_offsets = {}
            for field in cloud_msg.fields:
                field_offsets[field.name] = field.offset
                # Calculate field size in bytes (assuming FLOAT32=4, UINT32=4, etc.)
                field_size = 4  # Default for FLOAT32, UINT32
                if field.datatype == PointField.FLOAT64:
                    field_size = 8
                elif field.datatype == PointField.INT8 or field.datatype == PointField.UINT8:
                    field_size = 1
                elif field.datatype == PointField.INT16 or field.datatype == PointField.UINT16:
                    field_size = 2
                point_size = max(point_size, field.offset + field_size)

            # Validate buffer alignment
            if len(cloud_msg.data) % cloud_msg.point_step != 0:
                self.get_logger().warn(
                    f"PointCloud2 data size {len(cloud_msg.data)} not multiple of "
                    f"point_step {cloud_msg.point_step}. Truncating excess bytes."
                )
                valid_bytes = (len(cloud_msg.data) // cloud_msg.point_step) * cloud_msg.point_step
                cloud_data = cloud_msg.data[:valid_bytes]
            else:
                cloud_data = cloud_msg.data

            # Check if we have x,y,z fields
            required_fields = {'x', 'y', 'z'}
            if not required_fields.issubset(field_offsets.keys()):
                self.get_logger().error(
                    f"PointCloud2 missing required fields. Has: {field_offsets.keys()}"
                )
                return np.empty((0, 3), dtype=np.float32)

            # Create structured array
            dtype = np.dtype([
                ('x', np.float32),
                ('y', np.float32),
                ('z', np.float32)
            ])

            # Extract just x,y,z coordinates
            points = np.zeros(len(cloud_data) // cloud_msg.point_step, dtype=dtype)
            
            for i in range(points.shape[0]):
                start_idx = i * cloud_msg.point_step
                x_start = start_idx + field_offsets['x']
                y_start = start_idx + field_offsets['y']
                z_start = start_idx + field_offsets['z']
                
                points[i]['x'] = struct.unpack_from('f', cloud_data, x_start)[0]
                points[i]['y'] = struct.unpack_from('f', cloud_data, y_start)[0]
                points[i]['z'] = struct.unpack_from('f', cloud_data, z_start)[0]

            return np.column_stack((points['x'], points['y'], points['z']))

        except Exception as e:
            self.get_logger().error(f"PointCloud2 conversion error: {str(e)}")
            return np.empty((0, 3), dtype=np.float32)

    def transform_points(self, points, particle):
        """Transform points to particle's frame"""
        # Create rotation matrix from particle yaw
        rot = R.from_euler('z', particle.yaw).as_matrix()
        
        # Transform points
        translated = points - particle.position
        rotated = translated @ rot.T
        return rotated

def main(args=None):
    rclpy.init(args=args)
    slam_node = UnderwaterSLAM()
    
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        slam_node.get_logger().info("Shutting down SLAM node...")
    finally:
        slam_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()