#!/usr/bin/env python3
import rospy
import tf2_ros
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped

def curve(s, t):
    return np.array([0.6*np.cos(s), 0.6*np.sin(s), 1.0+0*s+0.2*np.cos(0.2*t)]).T

class PointCloudPublisher:
    def __init__(self):
        rospy.init_node("curve_pointcloud_publisher", anonymous=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.pointcloud_pub = rospy.Publisher("/curve_pointcloud", PointCloud2, queue_size=1)

        self.child_frame_id = "cf5"
        self.parent_frame_id = "world"  # Change if needed
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("Curve PointCloud publisher started")
        self.timer = rospy.Timer(rospy.Duration(0.01), self.publish_point_cloud)



    def create_pointcloud2(self, points, timestamp, frame_id):
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        point_cloud_msg = pc2.create_cloud_xyz32(header=rospy.Header(frame_id=frame_id, stamp=timestamp), points=points)
        return point_cloud_msg

    def publish_point_cloud(self, event):
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(self.parent_frame_id, self.child_frame_id, rospy.Time(0))
            
            # transform.header.stamp
            t = transform.header.stamp.to_sec()
            now = transform.header.stamp

            # Generate the curve
            s_values = np.linspace(0, 2*np.pi,1000)
            points = curve(s_values, t)

            # Convert to PointCloud2 message
            cloud_msg = self.create_pointcloud2(points, now, self.parent_frame_id)

            # Publish the point cloud
            self.pointcloud_pub.publish(cloud_msg)
        except tf2_ros.LookupException:
            rospy.logwarn(f"Could not lookup transform for {self.child_frame_id}.")
        except tf2_ros.ExtrapolationException:
            rospy.logwarn(f"Extrapolation error while looking up {self.child_frame_id} transform.")

if __name__ == "__main__":
    try:
        node = PointCloudPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
