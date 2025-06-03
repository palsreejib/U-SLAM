#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import pandas as pd
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class DVLPublisher(Node):
    def __init__(self):
        super().__init__('dvl_publisher')
        
        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('publish_rate', 10.0),
                ('dvl_frame', 'dvl_link'),
                ('base_frame', 'base_link'),
                ('file_path', '/home/sreejib/PAPER_2_Datasets/navigation.txt')
            ]
        )
        
        # Configuration
        self.publish_rate = self.get_parameter('publish_rate').value
        self.dvl_frame = self.get_parameter('dvl_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.file_path = self.get_parameter('file_path').value
        
        # QoS Setup
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        
        # Publishers
        self.vel_pub = self.create_publisher(TwistStamped, 'dvl/velocity', qos)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)
        
        # State
        self.clock = self.get_clock()
        self.index = 1
        self.dvl_data = self.load_dvl_data()
        
        if self.dvl_data is not None:
            self.timer = self.create_timer(1.0/self.publish_rate, self.publish_velocity)
            self.publish_diagnostics(0, "DVL Publisher initialized successfully")
        else:
            self.publish_diagnostics(2, "Failed to load DVL data")

    def load_dvl_data(self):
        """Load and preprocess DVL dataset with robust error handling"""
        try:
            df = pd.read_csv(
                self.file_path,
                delim_whitespace=True,
                skiprows=1,
                names=[
                    "Mission", "Date", "Time", "North", "East", "Heading",
                    "Roll", "Pitch", "Depth", "Altitude", "Speed"
                ]
            )
            
            # Validate columns
            required = ["Time", "North", "East", "Heading"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                self.get_logger().error(f"Missing columns: {missing}")
                return None
                
            # Convert and clean data
            df["Time"] = pd.to_numeric(
                df["Time"].astype(str).str.zfill(6).apply(
                    lambda x: int(x[:2])*3600 + int(x[2:4])*60 + int(x[4:6])
                ),
                errors='coerce'
            )
            
            numeric_cols = ["North", "East", "Heading"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Additional data validation
            df = df[(df['Time'] > 0) & 
                   (df['North'].abs() < 1000) & 
                   (df['East'].abs() < 1000) & 
                   (df['Heading'].between(0, 360))]
            df = df.dropna(subset=numeric_cols + ["Time"])
            
            if df.empty:
                self.get_logger().error("No valid data after preprocessing")
                return None
                
            self.get_logger().info(
                f"Loaded {len(df)} DVL entries. Time range: "
                f"{df['Time'].min()}s to {df['Time'].max()}s"
            )
            return df
            
        except Exception as e:
            self.get_logger().error(f"Data loading failed: {str(e)}")
            return None

    def compute_velocity(self, idx):
        """Robust velocity calculation with edge case handling"""
        try:
            if idx <= 0 or idx >= len(self.dvl_data):
                return 0.0, 0.0, 0.0
                
            prev = self.dvl_data.iloc[idx-1]
            curr = self.dvl_data.iloc[idx]
            
            dt = curr["Time"] - prev["Time"]
            if dt <= 0:
                return 0.0, 0.0, 0.0
                
            dx = curr["North"] - prev["North"]
            dy = curr["East"] - prev["East"]
            dyaw = np.radians(curr["Heading"] - prev["Heading"])
            
            return dx/dt, dy/dt, dyaw/dt
            
        except Exception as e:
            self.get_logger().warn(f"Velocity computation error: {str(e)}")
            return 0.0, 0.0, 0.0

    def publish_velocity(self):
        """Publish velocity and maintain TF tree"""
        if self.index >= len(self.dvl_data):
            self.get_logger().warn("End of dataset reached. Looping...")
            self.index = 1
            return
            
        # Compute and publish velocity
        vx, vy, vyaw = self.compute_velocity(self.index)
        
        msg = TwistStamped()
        msg.header.stamp = self.clock.now().to_msg()
        msg.header.frame_id = self.dvl_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.angular.z = vyaw
        
        # Publish transform (dvl_frame → base_link)
        transform = TransformStamped()
        transform.header = msg.header
        transform.child_frame_id = self.base_frame  # Child frame
        transform.header.frame_id = self.dvl_frame  # Parent frame
        
        # Realistic transform - DVL is typically mounted facing downward
        transform.transform.translation.z = -0.2  # 0.2m below base
        transform.transform.rotation.x = np.sqrt(2)/2  # 90 degree rotation around X-axis
        transform.transform.rotation.w = np.sqrt(2)/2
        
        self.vel_pub.publish(msg)
        self.tf_broadcaster.sendTransform(transform)
        
        self.get_logger().info(
            f"Published DVL velocity: [{vx:.2f}, {vy:.2f}, {vyaw:.2f}] m/s, rad/s",
            throttle_duration_sec=1.0
        )
        self.index += 1

    def publish_diagnostics(self, level, message):
        """Publish diagnostic information"""
        status = DiagnosticStatus()
        if level == 0:
            status.level = DiagnosticStatus.OK
        elif level == 1:
            status.level = DiagnosticStatus.WARN
        else:
            status.level = DiagnosticStatus.ERROR
        status.name = self.get_name()
        status.message = message
        
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()
        diag_msg.status.append(status)
        self.diag_pub.publish(diag_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DVLPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DVL publisher stopped")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()