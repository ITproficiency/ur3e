#!/usr/bin/env python3
"""
Script chạy 1 lần để lưu planning scene (bàn + chai nước) vào MoveIt Warehouse (SQLite).
Sau đó benchmark sẽ tự load scene này qua scene_name trong benchmark_ur3.yaml.

Cách dùng:
  ros2 run ur3_moveit_control save_scene_to_warehouse
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose
from moveit_msgs.srv import SaveMap
import time


SCENE_NAME = "table_and_bottle"

# ──────────────────────────────────────────────
# Kích thước và vị trí vật thể (đơn vị: mét)
# ──────────────────────────────────────────────
TABLE = dict(
    id="table",
    size=[0.8, 0.8, 0.02],          # dài x rộng x cao (mặt bàn mỏng)
    # Gốc robot ở mép bàn, mặt bàn cao 0.10 m so với base_link
    position=[0.45, 0.0, 0.10],
)

BOTTLE_HEIGHT = 0.22
BOTTLE = dict(
    id="water_bottle",
    radius=0.035,                    # bán kính chai
    height=BOTTLE_HEIGHT,
    # Chai đứng trên bàn: z = TABLE top + bottle_height/2
    position=[0.45, 0.0, TABLE["position"][2] + TABLE["size"][2] / 2 + BOTTLE_HEIGHT / 2],
)


def make_box(obj_id, size, pos, frame="base_link"):
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = obj_id
    co.operation = CollisionObject.ADD

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = size
    co.primitives.append(prim)

    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.w = 1.0
    co.primitive_poses.append(pose)
    return co


def make_cylinder(obj_id, radius, height, pos, frame="base_link"):
    co = CollisionObject()
    co.header.frame_id = frame
    co.id = obj_id
    co.operation = CollisionObject.ADD

    prim = SolidPrimitive()
    prim.type = SolidPrimitive.CYLINDER
    prim.dimensions = [height, radius]   # [height, radius] theo MoveIt convention
    co.primitives.append(prim)

    pose = Pose()
    pose.position.x = pos[0]
    pose.position.y = pos[1]
    pose.position.z = pos[2]
    pose.orientation.w = 1.0
    co.primitive_poses.append(pose)
    return co


class SceneSaver(Node):
    def __init__(self):
        super().__init__("scene_saver")
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)
        self.save_cli = self.create_client(SaveMap, "/warehouse/save_planning_scene")

    def run(self):
        # Publish scene lên MoveIt
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        scene_msg.name = SCENE_NAME

        scene_msg.world.collision_objects.append(
            make_box(TABLE["id"], TABLE["size"], TABLE["position"])
        )
        scene_msg.world.collision_objects.append(
            make_cylinder(BOTTLE["id"], BOTTLE["radius"], BOTTLE["height"], BOTTLE["position"])
        )

        self.get_logger().info(f"Publishing scene '{SCENE_NAME}' (ban + chai nuoc)...")
        for _ in range(5):
            self.scene_pub.publish(scene_msg)
            time.sleep(0.3)

        # Gọi service lưu vào warehouse
        if not self.save_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(
                "Service /warehouse/save_planning_scene khong co san.\n"
                "Dam bao move_group dang chay voi warehouse_plugin duoc cau hinh."
            )
            return

        req = SaveMap.Request()
        req.name = SCENE_NAME
        future = self.save_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is not None:
            self.get_logger().info(f"Da luu scene '{SCENE_NAME}' vao warehouse!")
        else:
            self.get_logger().error("Luu scene that bai.")


def main():
    rclpy.init()
    node = SceneSaver()
    node.run()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
