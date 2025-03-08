#!/usr/bin/env python3
import rospy
import tf
import tf2_ros
import geometry_msgs.msg
import tf2_msgs
import tf.transformations as tft
from math import pi

def transform_callback(msg, broadcaster):
    # Extract the original transform
    trans = msg.transform.translation
    rot = msg.transform.rotation
    
    # Convert quaternion to rotation matrix
    rot_matrix = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
    
    # Define the transformation matrix from Z-up to Y-up
    correction_matrix = tft.euler_matrix(-pi/2, 0, 0)  # Rotate -90° around X and -90° around Z
    corrected_matrix = correction_matrix.dot(rot_matrix)
    
    # Convert back to quaternion
    corrected_quat = tft.quaternion_from_matrix(corrected_matrix)
    
    # Apply translation correction
    corrected_trans = geometry_msgs.msg.Vector3()
    corrected_trans.x = trans.x
    corrected_trans.y = trans.z  # Swap Y and Z
    corrected_trans.z = -trans.y
    
    # Publish the corrected transform
    corrected_tf = geometry_msgs.msg.TransformStamped()
    corrected_tf.header.stamp = rospy.Time.now()
    corrected_tf.header.frame_id = msg.header.frame_id
    corrected_tf.child_frame_id = "foamball_corrected" #msg.child_frame_id
    corrected_tf.transform.translation = corrected_trans
    corrected_tf.transform.rotation.x = corrected_quat[0]
    corrected_tf.transform.rotation.y = corrected_quat[1]
    corrected_tf.transform.rotation.z = corrected_quat[2]
    corrected_tf.transform.rotation.w = corrected_quat[3]
    
    broadcaster.sendTransform(corrected_tf)

def main():
    rospy.init_node('tf_correction_node')
    tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tf_buffer)
    broadcaster = tf2_ros.TransformBroadcaster()
    
    rospy.Subscriber("/tf", tf2_msgs.msg.TFMessage, 
                     lambda msg: [transform_callback(t, broadcaster) for t in msg.transforms if t.child_frame_id == "foamball/base_link"])
    
    rospy.spin()

if __name__ == '__main__':
    main()
