#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped

class FoamBallPointCloudNode:
    def __init__(self):
        rospy.init_node('foam_ball_pcl_node', anonymous=True)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pcl_pub = rospy.Publisher('/foamball/pointcloud', PointCloud2, queue_size=1)
        
        self.sphere_radius = 0.06  # Radius of the sphere (0.12m diameter)
        self.num_points = 20  # Number of points in the point cloud
        
        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_point_cloud)

    def generate_sphere_points(self, center):
        """Generates a point cloud representing a sphere around the given center."""
        phi = np.linspace(0, np.pi, self.num_points)
        theta = np.linspace(0, 2 * np.pi, self.num_points)

        phi, theta = np.meshgrid(phi, theta)
        phi, theta = phi.ravel(), theta.ravel()
        
        x = center[0] + self.sphere_radius * np.sin(phi) * np.cos(theta)
        y = center[1] + self.sphere_radius * np.sin(phi) * np.sin(theta)
        z = center[2] + self.sphere_radius * np.cos(phi)
        
        points = zip(x, y, z)
        return list(points)

    def publish_point_cloud(self, event):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform('world', 'foamball_corrected', rospy.Time(0))
            position = transform.transform.translation
            
            points = self.generate_sphere_points((position.x, position.y, position.z))
            
            header = rospy.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]
            
            cloud = pc2.create_cloud(header, fields, points)
            self.pcl_pub.publish(cloud)
        except tf2_ros.LookupException:
            rospy.logwarn("Could not lookup transform for foamball_corrected.")
        except tf2_ros.ExtrapolationException:
            rospy.logwarn("Extrapolation error while looking up foamball transform.")

if __name__ == '__main__':
    try:
        node = FoamBallPointCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
