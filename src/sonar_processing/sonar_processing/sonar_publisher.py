import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
import numpy as np
import struct
import os
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from cv_bridge import CvBridge

class SonarPublisher(Node):
    def __init__(self):
        super().__init__('sonar_publisher')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('sonar_frame', 'sonar_link'),
                ('base_frame', 'base_link'),
                ('data_dir', '/media/sreejib/Crucial X6/PAPER_2/AURORA/AURORA-JC125-M87-side-scan-sonar/side-scan-sonar/xtf/xtf-navigation'),
                ('publish_rate', 1.0)
            ]
        )
        
        # Configuration
        self.sonar_frame = self.get_parameter('sonar_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.data_dir = self.get_parameter('data_dir').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # QoS Setup
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Publishers
        self.publisher = self.create_publisher(PointCloud2, 'sonar/pointcloud', qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Image publisher for debug
        self.bridge = CvBridge()
        self.debug_pub = self.create_publisher(Image, 'sonar/debug_image', 1)
        
        # Publish static transform
        self.publish_static_transform()
        
        # Load data
        self.xtf_files = self.find_xtf_files()
        if not self.xtf_files:
            self.publish_diagnostics(2, "No XTF files found!")
            rclpy.shutdown()
            return
            
        self.current_file_index = 0
        self.current_data = self.load_xtf_data(self.xtf_files[self.current_file_index])
        
        if self.current_data is not None:
            self.timer = self.create_timer(1.0/self.publish_rate, self.publish_sonar)
            self.publish_diagnostics(0, "Sonar Publisher initialized successfully")
        else:
            self.publish_diagnostics(2, "Failed to load sonar data")

    def publish_static_transform(self):
        """Publish a static transform from base_link to sonar_frame"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base_frame  # Parent frame
        transform.child_frame_id = self.sonar_frame  # Child frame
        
        # Typical sonar mounting position (adjust for your setup)
        transform.transform.translation.x = 0.5   # 0.5m forward from base
        transform.transform.translation.z = -0.3  # 0.3m below base
        transform.transform.rotation.w = 1.0      # No rotation
        
        self.tf_broadcaster.sendTransform(transform)
        self.get_logger().info("Published static transform: base_link → sonar_frame")

    def find_xtf_files(self):
        """Find all .xtf files in the data directory"""
        if not os.path.exists(self.data_dir):
            self.get_logger().error(f"Directory not found: {self.data_dir}")
            return []
            
        return sorted([os.path.join(self.data_dir, f) 
                      for f in os.listdir(self.data_dir) 
                      if f.endswith('.xtf')])

    def load_xtf_data(self, file_path):
        """Load sonar data from an XTF file"""
        self.get_logger().info(f"Loading XTF file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            points = self.extract_sonar_points(data)
            if points is None:
                raise ValueError("Failed to extract sonar points")
            
            self.get_logger().info(f"Loaded {len(points)} points from {file_path}")
            return np.array(points, dtype=np.float32)

        except Exception as e:
            self.get_logger().error(f"Failed to read XTF file {file_path}: {e}")
            return None

    def extract_sonar_points(self, data):
        """Extract sonar points from binary data with visualization"""
        points = []
        intensity = []
        MAX_POINTS = 5000  # Limit points for performance
        
        try:
            # Create debug image
            debug_img = np.zeros((500, 500, 3), dtype=np.uint8)
            
            index = 0
            while index < len(data) and len(points) < MAX_POINTS:
                # Parse sonar packet (simplified - adjust for your format)
                if index + 16 > len(data):
                    break
                    
                # Extract point data (example format)
                range_val = struct.unpack_from('f', data, index)[0]
                angle = struct.unpack_from('f', data, index+4)[0]
                intensity_val = struct.unpack_from('H', data, index+8)[0]
                
                # Convert polar to Cartesian
                x = range_val * np.cos(angle)
                y = range_val * np.sin(angle)
                z = 0  # 2D sonar
                
                points.append([x, y, z])
                intensity.append(intensity_val / 65535.0)  # Normalize
                
                # Add to debug image
                img_x = int(250 + x * 10)
                img_y = int(250 + y * 10)
                if 0 <= img_x < 500 and 0 <= img_y < 500:
                    color_val = int(intensity_val / 256)
                    debug_img[img_y, img_x] = (0, color_val, color_val)
                
                index += 16
                
            # Publish debug image
            debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            self.debug_pub.publish(debug_msg)
            
            return points
            
        except Exception as e:
            self.get_logger().error(f"Error parsing sonar data: {e}")
            return None

    def publish_sonar(self):
        """Publish sonar data as PointCloud2"""
        if self.current_data is None or len(self.current_data) == 0:
            self.get_logger().warn("No valid SONAR data to publish")
            return
            
        # Create PointCloud2 message
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.sonar_frame
        msg.height = 1
        msg.width = len(self.current_data)
        
        # Fields: x, y, z, intensity
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        msg.is_bigendian = False
        msg.point_step = 16  # 4 fields * 4 bytes each
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        
        # Create point cloud data with intensity
        cloud_data = []
        for point in self.current_data:
            cloud_data.extend([point[0], point[1], point[2], 1.0])  # intensity=1.0
            
        msg.data = np.array(cloud_data, dtype=np.float32).tobytes()
        
        # Publish
        self.publisher.publish(msg)
        self.get_logger().info(f"Published {len(self.current_data)} sonar points")
        
        # Move to next file
        self.current_file_index = (self.current_file_index + 1) % len(self.xtf_files)
        self.current_data = self.load_xtf_data(self.xtf_files[self.current_file_index])

    def publish_diagnostics(self, level, message):
        """Publish diagnostic information"""
        status = DiagnosticStatus()
        status.level = level  # 0=OK, 1=WARN, 2=ERROR
        status.name = self.get_name()
        status.message = message
        
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()
        diag_msg.status.append(status)
        self.diag_pub.publish(diag_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SonarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()