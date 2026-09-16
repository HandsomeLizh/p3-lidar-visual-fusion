import time,json,rclpy
from rclpy.node import Node
rclpy.init()
n=Node('fusion_readonly_imu_audit')
end=time.monotonic()+4
while time.monotonic()<end:rclpy.spin_once(n,timeout_sec=.1)
topics=n.get_topic_names_and_types()
print(json.dumps({'domain':10,'sensor_topics':[(t,ty) for t,ty in topics if any(x in t.lower() for x in ['imu','wit','os1','fusion']) or any('Imu' in v for v in ty)]},indent=2))
n.destroy_node();rclpy.shutdown()
