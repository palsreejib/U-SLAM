#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
import cv2
from cv_bridge import CvBridge
import glob
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_frame', 'camera_optical_frame'),
                ('base_frame', 'base_link'),
                ('map_frame', 'map'),
                ('data_dir', '/media/sreejib/Crucial X6/PAPER_2/AURORA/AURORA-JC125-M87-vertical-camera/vertical-camera/jpg'),
                ('publish_rate', 10.0)
            ]
        )
        
        # Configuration
        self.camera_frame = self.get_parameter('camera_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.data_dir = self.get_parameter('data_dir').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # QoS Setup
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Initialize components
        self.bridge = CvBridge()
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.dynamic_tf_broadcaster = TransformBroadcaster(self)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        
        # State
        self.index = 0
        self.camera_pose = np.zeros(3)  # [x, y, theta]
        self.prev_image = None
        
        # Feature tracking
        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Setup publishers
        self.image_pub = self.create_publisher(Image, 'camera/image_raw', qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera/camera_info', qos)
        self.pose_pub = self.create_publisher(PoseStamped, 'camera/pose', qos)
        
        # Load camera calibration
        self.camera_info = self.load_camera_calibration()
        
        # Find image files
        self.image_files = sorted(glob.glob(f'{self.data_dir}/*.jpg'))
        if not self.image_files:
            self.publish_diagnostics(2, "No image files found!")
            self.get_logger().error("No image files found in directory: " + self.data_dir)
            return  # Don't shutdown here, let the node fail gracefully
            
        self.get_logger().info(f"Found {len(self.image_files)} images")
        
        # Publish initial transforms
        self.publish_base_transform()
        self.publish_camera_transform()
        
        # Timer
        self.timer = self.create_timer(1.0/self.publish_rate, self.publish_data)
        self.publish_diagnostics(0, "Camera Publisher initialized successfully")

    def load_camera_calibration(self):
        """Generate default camera calibration"""
        info = CameraInfo()
        info.header.frame_id = self.camera_frame
        info.width = 640  # Adjust based on your camera
        info.height = 480
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]  # Distortion coefficients
        info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]  # Intrinsic matrix
        return info

    def publish_base_transform(self):
        """Publish static transform from map to base_link"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.map_frame
        transform.child_frame_id = self.base_frame
        transform.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform(transform)

    def publish_camera_transform(self):
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()  # Critical!
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "camera_optical_frame"
        
        # Example camera position (adjust for your robot)
        transform.transform.translation.x = 0.1  # 10cm forward
        transform.transform.translation.z = 0.05 # 5cm up
        
        # No rotation (or adjust if needed)
        transform.transform.rotation.w = 1.0
        
        self.static_tf_broadcaster.sendTransform(transform)

    def estimate_movement(self, current_img):
        """Estimate camera movement using optical flow"""
        if self.prev_image is None:
            self.prev_image = current_img
            return 0, 0, 0
            
        # Convert to grayscale
        prev_gray = cv2.cvtColor(self.prev_image, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, 
                                        qualityLevel=0.3, minDistance=7)
        if prev_pts is None:
            return 0, 0, 0
            
        # Calculate optical flow
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, 
                                                    prev_pts, None)
        
        # Filter valid points
        idx = np.where(status==1)[0]
        if len(idx) < 10:
            return 0, 0, 0
            
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        # Calculate homography
        H, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        if H is None:
            return 0, 0, 0
            
        # Extract translation and rotation
        dx = H[0,2] * 0.01  # Scale factor
        dy = H[1,2] * 0.01
        rotation = np.arctan2(H[1,0], H[0,0])
        
        self.prev_image = current_img
        return dx, dy, rotation

    def publish_data(self):
        try:
            # Load current image
            img_path = self.image_files[self.index % len(self.image_files)]
            current_img = cv2.imread(img_path)
            if current_img is None:
                self.get_logger().error(f"Failed to load image: {img_path}")
                return
                
            # Estimate camera movement
            dx, dy, dtheta = self.estimate_movement(current_img)
            self.camera_pose += np.array([dx, dy, dtheta])
            
            # Prepare messages
            stamp = self.get_clock().now().to_msg()
            
            # Camera pose in map frame
            cam_pose = PoseStamped()
            cam_pose.header.stamp = stamp
            cam_pose.header.frame_id = self.map_frame
            cam_pose.pose.position.x = self.camera_pose[0]
            cam_pose.pose.position.y = self.camera_pose[1]
            
            # Convert theta to quaternion (using scipy's Rotation)
            q = R.from_euler('z', self.camera_pose[2]).as_quat()
            cam_pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            
            # Image message
            img_msg = self.bridge.cv2_to_imgmsg(current_img, encoding='bgr8')
            img_msg.header.stamp = stamp
            img_msg.header.frame_id = self.camera_frame
            
            # Camera info
            self.camera_info.header.stamp = stamp
            
            # Publish everything
            self.pose_pub.publish(cam_pose)
            self.image_pub.publish(img_msg)
            self.camera_info_pub.publish(self.camera_info)
            
            self.index += 1
            self.get_logger().info(
                f"Published frame {self.index}: Position=({self.camera_pose[0]:.2f}, {self.camera_pose[1]:.2f})",
                throttle_duration_sec=1.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Error in publish_data: {str(e)}")
            self.publish_diagnostics(2, f"Publish error: {str(e)}")

    def publish_diagnostics(self, level, message):
        """Publish diagnostic information"""
        status = DiagnosticStatus()
        status.level = level  # Convert integer to bytes
        status.name = self.get_name()
        status.message = message
        
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()
        diag_msg.status.append(status)
        self.diag_pub.publish(diag_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down due to keyboard interrupt")
    except Exception as e:
        node.get_logger().error(f"Node error: {str(e)}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()