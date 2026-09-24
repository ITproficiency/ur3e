#!/usr/bin/env python3

import argparse
import math
import os
import random
from pathlib import Path
import sys

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.utilities import remove_ros_args

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from ur3_moveit_control.action import UR3Control
from grasp_profile import (
    APPROACH_CLEARANCE as FIXED_APPROACH_CLEARANCE,
    BOTTLE_X as FIXED_BOTTLE_X,
    BOTTLE_Y as FIXED_BOTTLE_Y,
    BOTTLE_Z as FIXED_BOTTLE_Z,
    CLOSE_POSITION as FIXED_CLOSE_POSITION,
    GRASP_DEPTH_OFFSET as FIXED_GRASP_DEPTH_OFFSET,
    LIFT_OFFSET as FIXED_LIFT_OFFSET,
    MAX_EFFORT as FIXED_MAX_EFFORT,
    OPEN_POSITION as FIXED_OPEN_POSITION,
    POUR_WRIST_ANGLE_DEG as FIXED_POUR_WRIST_ANGLE_DEG,
    TCP_GRASP_OFFSET as FIXED_TCP_GRASP_OFFSET,
    TRANSPORT_X_MAX as FIXED_TRANSPORT_X_MAX,
    TRANSPORT_Y_MAX as FIXED_TRANSPORT_Y_MAX,
    TRANSPORT_Y_MIN as FIXED_TRANSPORT_Y_MIN,
    append_successful_waypoint,
    pouring_joint_goal,
    radial_side_grasp_geometry,
)

ARM_JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

def wait_for_user(step_name: str):
    print("\n" + "=" * 50)
    input(f">> Bấm [ENTER] để chạy: [{step_name}]...")
    print("=" * 50 + "\n")

class UR3ActionClient(Node):

    GRIPPER_MIN_POSITION = 0.0
    # Keep a small margin from the simulated prismatic joint's hard stop.
    GRIPPER_MAX_POSITION = 0.06

    def __init__(self):
        super().__init__('ur3_action_client')

        # Action client for controlling the UR3 arm
        self._arm_action_client = ActionClient(
            self,
            UR3Control,
            'ur3_control'
        )

        # Action client for controlling the gripper
        self._gripper_action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/gripper_controller/follow_joint_trajectory'
        )
        self._pending_lift_offset = None
        self._gripper_completion_timer = None
        self._pick_sequence = None
        self._latest_gripper_effort = None
        self._peak_gripper_effort = 0.0
        self._latest_joint_positions = {}
        self._joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

    def joint_state_callback(self, msg):
        """Track measured gripper effort reported by ros2_control."""

        self._latest_joint_positions = dict(zip(msg.name, msg.position))

        try:
            joint_index = msg.name.index('gripper_joint')
        except ValueError:
            return

        if joint_index >= len(msg.effort):
            return

        measured_effort = float(msg.effort[joint_index])
        self._latest_gripper_effort = measured_effort
        self._peak_gripper_effort = max(
            self._peak_gripper_effort,
            abs(measured_effort)
        )

    def capture_pregrasp_joints(self):
        """Keep the current six arm joints in memory until the trial passes."""

        missing = [
            name for name in ARM_JOINT_NAMES
            if name not in self._latest_joint_positions
        ]
        if missing:
            self.get_logger().warning(
                'Cannot capture pre-grasp waypoint; missing joint states: '
                + ', '.join(missing)
            )
            return

        joints = [
            self._latest_joint_positions[name]
            for name in ARM_JOINT_NAMES
        ]
        self._pick_sequence['captured_pregrasp_joints'] = joints
        self.get_logger().info(
            'Captured pre-grasp joints in memory: '
            + '[' + ', '.join(f'{value:.9f}' for value in joints) + ']'
        )

    def save_successful_waypoint(self):
        """Append the captured pre-grasp waypoint after full trial success."""

        joints = self._pick_sequence.get('captured_pregrasp_joints')
        if joints is None:
            self.get_logger().warning(
                'Trial succeeded, but no pre-grasp joint snapshot was captured'
            )
            return

        qx, qy, qz, qw = self._pick_sequence['grasp_quaternion']
        row = {
            'source': 'send_goal',
            'ros_time_ns': self.get_clock().now().nanoseconds,
            'object_x': self._pick_sequence['object_x'],
            'object_y': self._pick_sequence['object_y'],
            'object_z': self._pick_sequence['object_z'],
            'pregrasp_x': self._pick_sequence['approach_x'],
            'pregrasp_y': self._pick_sequence['approach_y'],
            'pregrasp_z': self._pick_sequence['grasp_z'],
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'qw': qw,
            **dict(zip(ARM_JOINT_NAMES, joints)),
            'transport_dx': self._pick_sequence['transport_dx'],
            'transport_dy': self._pick_sequence['transport_dy'],
        }

        path = append_successful_waypoint(row)

        self.get_logger().info(
            f'Saved successful pre-grasp waypoint to {path}'
        )

    # =========================================================
    # ARM COMMANDS
    # =========================================================

    def send_pose_goal(self, x, y, z, qx, qy, qz, qw):
        """Send a Cartesian pose goal to the UR3 arm."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_POSE

        goal_msg.pose_goal.header.frame_id = 'base_link'
        goal_msg.pose_goal.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal_msg.pose_goal.pose.position.x = float(x)
        goal_msg.pose_goal.pose.position.y = float(y)
        goal_msg.pose_goal.pose.position.z = float(z)

        goal_msg.pose_goal.pose.orientation.x = float(qx)
        goal_msg.pose_goal.pose.orientation.y = float(qy)
        goal_msg.pose_goal.pose.orientation.z = float(qz)
        goal_msg.pose_goal.pose.orientation.w = float(qw)

        if not self._arm_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The UR3 action server is not available'
            )
            return

        self.get_logger().info(
            f'Sending pose goal: '
            f'x={x:.3f}, y={y:.3f}, z={z:.3f}, '
            f'quaternion=({qx:.3f}, {qy:.3f}, '
            f'{qz:.3f}, {qw:.3f})'
        )

        future = self._arm_action_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            self.arm_goal_response_callback
        )

    def send_home_goal(self):
        """Send the robot to its predefined home configuration."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_HOME

        if not self._arm_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The UR3 action server is not available'
            )
            return

        self.get_logger().info('Sending home goal...')

        future = self._arm_action_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            self.arm_goal_response_callback
        )

    def prepare_next_trial(self):
        """Clear stale Gazebo and MoveIt attachment state before grasping."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.PREPARE_NEXT_TRIAL

        if not self._arm_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The UR3 action server is not available'
            )
            return

        self.get_logger().info(
            'Preparing reusable pick_box state before the grasp sequence...'
        )
        future = self._arm_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.arm_goal_response_callback)

    def send_joint_goal(self, j1, j2, j3, j4, j5, j6):
        """Send a six-joint position goal to the UR3 arm."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_JOINT

        goal_msg.joint_goal.position = [
            float(j1),
            float(j2),
            float(j3),
            float(j4),
            float(j5),
            float(j6)
        ]

        if not self._arm_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The UR3 action server is not available'
            )
            return

        self.get_logger().info(
            f'Sending joint goal: '
            f'[{j1}, {j2}, {j3}, {j4}, {j5}, {j6}]'
        )

        future = self._arm_action_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            self.arm_goal_response_callback
        )

    def attach_and_lift(
            self, z_offset, object_x=0.62, object_y=0.0, object_z=0.05):
        """Attach pick_box to the gripper, then lift it along base_link Z."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.ATTACH_AND_LIFT
        goal_msg.cartesian_z_offset = float(z_offset)
        goal_msg.object_x = float(object_x)
        goal_msg.object_y = float(object_y)
        goal_msg.object_z = float(object_z)

        if not self._arm_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The UR3 action server is not available'
            )
            return

        self.get_logger().info(
            f'Attaching pick_box and requesting Cartesian lift: '
            f'object=({object_x:.3f}, {object_y:.3f}, {object_z:.3f}), '
            f'dz={z_offset:+.3f} m'
        )

        future = self._arm_action_client.send_goal_async(
            goal_msg
        )

        future.add_done_callback(
            self.arm_goal_response_callback
        )

    def grasp_bottle_at(
            self,
            object_x,
            object_y,
            object_z,
            approach_clearance=FIXED_APPROACH_CLEARANCE,
            tcp_grasp_offset=FIXED_TCP_GRASP_OFFSET,
            grasp_depth_offset=FIXED_GRASP_DEPTH_OFFSET,
            lift_offset=FIXED_LIFT_OFFSET,
            open_position=FIXED_OPEN_POSITION,
            close_position=FIXED_CLOSE_POSITION,
            max_effort=FIXED_MAX_EFFORT,
            pour_angle_deg=FIXED_POUR_WRIST_ANGLE_DEG,
            transport_dx=0.1,
            transport_dy=0.5):
        """Run the shared side-grasp flow for an object pose in base_link."""

        # Check connectivity before creating the asynchronous state machine.
        # Without this preflight, a missing server left the process spinning
        # forever even though no goal had been sent.
        if not self._arm_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(
                'The UR3 action server is not available. '
                f'Client ROS_DOMAIN_ID={os.environ.get("ROS_DOMAIN_ID", "0")}, '
                'ROS_LOCALHOST_ONLY='
                f'{os.environ.get("ROS_LOCALHOST_ONLY", "0")}. '
                'Gazebo, MoveIt, ur3_control_node, and this client must use '
                'the same ROS discovery environment.'
            )
            return False

        grasp_z = float(object_z) + float(tcp_grasp_offset)
        geometry = radial_side_grasp_geometry(
            float(object_x),
            float(object_y),
            float(approach_clearance),
            float(grasp_depth_offset),
        )
        if transport_dx is None:
            transport_dx = random.uniform(
                -FIXED_TRANSPORT_X_MAX,
                FIXED_TRANSPORT_X_MAX,
            )
        if transport_dy is None:
            transport_side = random.choice((-1.0, 1.0))
            transport_dy = transport_side * random.uniform(
                FIXED_TRANSPORT_Y_MIN,
                FIXED_TRANSPORT_Y_MAX,
            )
        self._pick_sequence = {
            'stage': 'preparing',
            'object_x': float(object_x),
            'object_y': float(object_y),
            'object_z': float(object_z),
            'grasp_x': geometry['grasp_x'],
            'grasp_y': geometry['grasp_y'],
            'grasp_z': grasp_z,
            'approach_x': geometry['approach_x'],
            'approach_y': geometry['approach_y'],
            'advance_x': geometry['advance_x'],
            'advance_y': geometry['advance_y'],
            'grasp_quaternion': geometry['quaternion'],
            'transport_dx': transport_dx,
            'transport_dy': transport_dy,
            'lift_offset': float(lift_offset),
            'open_position': float(open_position),
            'close_position': float(close_position),
            'max_effort': float(max_effort),
            'pour_angle_deg': float(pour_angle_deg),
        }

        self.get_logger().info(
            'Starting parameterized bottle side-grasp in base_link: '
            f'object=({object_x:.3f}, {object_y:.3f}, {object_z:.3f}), '
            f'pre_grasp=({geometry["approach_x"]:.3f}, '
            f'{geometry["approach_y"]:.3f}, {grasp_z:.3f}), '
            f'grasp_tcp=({geometry["grasp_x"]:.3f}, '
            f'{geometry["grasp_y"]:.3f}, {grasp_z:.3f}), '
            f'approach_yaw={math.degrees(geometry["yaw"]):+.1f} deg, '
            f'post_lift_cartesian=(dx={transport_dx:+.3f}, '
            f'dy={transport_dy:+.3f}, dz=+0.000)'
        )
        self.get_logger().info(
            'Bottle grasp setup 1/2: clearing stale attachment state'
        )
        self.prepare_next_trial()
        return True

    def grasp_fixed_bottle(
            self,
            approach_clearance=FIXED_APPROACH_CLEARANCE,
            tcp_grasp_offset=FIXED_TCP_GRASP_OFFSET,
            grasp_depth_offset=FIXED_GRASP_DEPTH_OFFSET,
            lift_offset=FIXED_LIFT_OFFSET,
            open_position=FIXED_OPEN_POSITION,
            close_position=FIXED_CLOSE_POSITION,
            max_effort=FIXED_MAX_EFFORT,
            pour_angle_deg=FIXED_POUR_WRIST_ANGLE_DEG):
        """Backward-compatible wrapper for the launch file's fixed bottle."""

        return self.grasp_bottle_at(
            FIXED_BOTTLE_X,
            FIXED_BOTTLE_Y,
            FIXED_BOTTLE_Z,
            approach_clearance=approach_clearance,
            tcp_grasp_offset=tcp_grasp_offset,
            grasp_depth_offset=grasp_depth_offset,
            lift_offset=lift_offset,
            open_position=open_position,
            close_position=close_position,
            max_effort=max_effort,
            pour_angle_deg=pour_angle_deg,
        )

    def rotate_gripper_for_pour(self):
        """Plan a wrist-only pouring tilt with the bottle still attached."""

        missing = [
            name for name in ARM_JOINT_NAMES
            if name not in self._latest_joint_positions
        ]
        if missing:
            self.abort_pick_sequence(
                'pour_wrist missing current joint states: '
                + ', '.join(missing)
            )
            return

        current_joints = [
            self._latest_joint_positions[name] for name in ARM_JOINT_NAMES
        ]
        try:
            target_joints, applied_delta = pouring_joint_goal(
                current_joints,
                self._pick_sequence['pour_angle_deg'],
            )
        except ValueError as error:
            self.abort_pick_sequence(f'pour_wrist failed: {error}')
            return

        self.get_logger().info(
            'Planning bottle-pouring tilt using wrist_3_joint only: '
            f'requested={self._pick_sequence["pour_angle_deg"]:+.1f} deg, '
            f'applied={math.degrees(applied_delta):+.1f} deg, '
            f'target={target_joints[5]:+.3f} rad'
        )
        self.send_joint_goal(*target_joints)

    def abort_pick_sequence(self, reason):
        """Stop the combined sequence after a failed stage."""

        self.get_logger().error(f'Bottle grasp aborted: {reason}')
        self._pick_sequence = None
        rclpy.shutdown()

    def move_cartesian(self, x_offset, y_offset, z_offset, strict=False):
        """Move gripper_tcp by an XYZ offset in the base_link frame."""

        x_offset = float(x_offset)
        y_offset = float(y_offset)
        z_offset = float(z_offset)
        if max(abs(x_offset), abs(y_offset), abs(z_offset)) < 1e-6:
            self.get_logger().error('At least one Cartesian offset must be non-zero')
            return

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = (
            UR3Control.Goal.MOVE_CARTESIAN_STRICT
            if strict else UR3Control.Goal.MOVE_CARTESIAN
        )
        goal_msg.cartesian_x_offset = x_offset
        goal_msg.cartesian_y_offset = y_offset
        goal_msg.cartesian_z_offset = z_offset

        if not self._arm_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('The UR3 action server is not available')
            return

        motion_kind = 'strict Cartesian' if strict else 'Cartesian'
        self.get_logger().info(
            f'Requesting {motion_kind} motion from current gripper pose: '
            f'dx={x_offset:+.3f}, dy={y_offset:+.3f}, '
            f'dz={z_offset:+.3f} m'
        )
        future = self._arm_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.arm_goal_response_callback)

    def move_to_xy(self, target_x, target_y):
        """Move to absolute base_link X/Y while keeping current Z and orientation."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.MOVE_TO_XY
        goal_msg.target_x = float(target_x)
        goal_msg.target_y = float(target_y)

        if not self._arm_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('The UR3 action server is not available')
            return

        self.get_logger().info(
            f'Requesting absolute XY target in base_link: '
            f'x={target_x:+.3f}, y={target_y:+.3f}; keeping current Z'
        )
        future = self._arm_action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.arm_goal_response_callback)

    def detach_object(self, open_position=0.06, max_effort=100.0):
        """Detach pick_box, then automatically open the gripper."""

        goal_msg = UR3Control.Goal()
        goal_msg.command_type = UR3Control.Goal.DETACH_OBJECT

        if not self._arm_action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('The UR3 action server is not available')
            return

        self.get_logger().info(
            'Requesting pick_box detach; gripper will open afterward...'
        )
        future = self._arm_action_client.send_goal_async(goal_msg)
        future.add_done_callback(
            lambda response_future: self.detach_goal_response_callback(
                response_future,
                float(open_position),
                float(max_effort)
            )
        )

    def detach_goal_response_callback(
            self, future, open_position, max_effort):
        """Wait specifically for detach, then arrange automatic opening."""

        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(f'Failed to send detach goal: {error}')
            return

        if not goal_handle.accepted:
            self.get_logger().error('The detach goal was rejected')
            return

        self.get_logger().info('The detach goal was accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda detach_future: self.detach_result_callback(
                detach_future,
                open_position,
                max_effort
            )
        )

    def detach_result_callback(self, future, open_position, max_effort):
        """Open the gripper only after the detach goal has completed."""

        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as error:
            self.get_logger().error(f'Failed to receive detach result: {error}')
            return

        self.get_logger().info(
            f'Detach action completed: success={result.success}, '
            f'message="{result.message}"'
        )
        if (
            self._pick_sequence is not None
            and self._pick_sequence['stage'] == 'detaching'
        ):
            self._pick_sequence['stage'] = 'releasing'
        self.get_logger().info(
            'Detach action finished; opening the gripper automatically'
        )
        self.open_gripper(open_position, max_effort)

    def arm_goal_response_callback(self, future):
        """Process the response after sending an arm goal."""

        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Failed to send the arm goal: {error}'
            )
            return

        if not goal_handle.accepted:
            self.get_logger().error(
                'The arm goal was rejected'
            )
            return

        self.get_logger().info(
            'The arm goal was accepted'
        )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.arm_result_callback
        )

    def arm_result_callback(self, future):
        """Process the final arm result."""

        try:
            wrapped_result = future.result()
            result = wrapped_result.result
        except Exception as error:
            self.get_logger().error(
                f'Failed to receive the arm result: {error}'
            )
            return

        self.get_logger().info(
            f'Arm action completed: '
            f'success={result.success}, '
            f'message="{result.message}"'
        )

        if self._pick_sequence is not None:
            if not result.success:
                self.abort_pick_sequence(
                    f'arm stage {self._pick_sequence["stage"]} failed: '
                    f'{result.message}'
                )
                return

            stage = self._pick_sequence['stage']
            if stage == 'preparing':
                self._pick_sequence['stage'] = 'homing'
                self.get_logger().info(
                    'Bottle grasp setup 2/2: moving to the stable home '
                    'configuration before pre-grasp planning'
                )
                self.send_home_goal()
                return

            if stage == 'homing':
                self._pick_sequence['stage'] = 'opening'
                self.get_logger().info(
                    'Home completed; opening the gripper before planning '
                    'to pre-grasp'
                )
                self.open_gripper(
                    self._pick_sequence['open_position'],
                    self._pick_sequence['max_effort'])
                return

            if stage == 'pregrasp':
                self.capture_pregrasp_joints()
                self._pick_sequence['stage'] = 'advancing'
                self.get_logger().info(
                    'Bottle grasp stage 2/6: advancing horizontally to the bottle'
                )
                self.move_cartesian(
                    self._pick_sequence['advance_x'],
                    self._pick_sequence['advance_y'],
                    0.0,
                    strict=True)
                return

            if stage == 'advancing':
                self._pick_sequence['stage'] = 'closing'
                self.get_logger().info(
                    'Bottle grasp stage 3/6: closing both gripper fingers'
                )
                self.close_gripper(
                    self._pick_sequence['close_position'],
                    self._pick_sequence['max_effort'])
                return

            if stage == 'lifting':
                self._pick_sequence['stage'] = 'transporting'
                self.get_logger().info(
                    'Bottle grasp stage 5/6: moving the lifted bottle sideways '
                    'with a strict Cartesian path'
                )
                self.move_cartesian(
                    self._pick_sequence['transport_dx'],
                    self._pick_sequence['transport_dy'],
                    0.0,
                    strict=True)
                return

            if stage == 'transporting':
                self._pick_sequence['stage'] = 'pouring'
                self.get_logger().info(
                    'Bottle grasp stage 6/6: transport completed; rotating '
                    'the gripper into '
                    'the pouring pose while keeping the bottle attached'
                )
                self.rotate_gripper_for_pour()
                return

            if stage == 'pouring':
                self.get_logger().info(
                    'Bottle side-grasp, transport, and pouring rotation '
                    'completed successfully; bottle remains attached and '
                    'the gripper remains closed'
                )
                self.save_successful_waypoint()
                self._pick_sequence = None
                rclpy.shutdown()
                return

        rclpy.shutdown()

    # =========================================================
    # GRIPPER COMMANDS
    # =========================================================

    def send_gripper_goal(self, position, max_effort):
        """
        Send a position goal to the gripper.

        Args:
            position:
                Desired gripper position.

            max_effort:
                Maximum allowed gripper effort.
        """

        self._latest_gripper_effort = None
        self._peak_gripper_effort = 0.0

        if not self._gripper_action_client.wait_for_server(
            timeout_sec=5.0
        ):
            self.get_logger().error(
                'The gripper action server is not available'
            )
            return

        requested_position = float(position)
        safe_position = min(
            max(requested_position, self.GRIPPER_MIN_POSITION),
            self.GRIPPER_MAX_POSITION
        )
        if safe_position != requested_position:
            self.get_logger().warning(
                f'Clamping gripper position from {requested_position:.4f} '
                f'to safe range value {safe_position:.4f}'
            )

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = [
            'gripper_joint',
            'gripper_joint_mimic'
        ]
        point = JointTrajectoryPoint()
        point.positions = [safe_position, safe_position]
        point.time_from_start = Duration(sec=2, nanosec=0)
        goal_msg.trajectory.points = [point]

        self.get_logger().info(
            f'Sending gripper goal: '
            f'left={safe_position:.4f}, right={safe_position:.4f}, '
            f'requested_max_effort={max_effort:.2f}'
        )

        future = self._gripper_action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.gripper_feedback_callback
        )

        future.add_done_callback(
            self.gripper_goal_response_callback
        )

    def open_gripper(self, position, max_effort):
        """
        Open the gripper using the supplied position and effort.
        """

        self.send_gripper_goal(
            position=position,
            max_effort=max_effort
        )

    def close_gripper(self, position, max_effort):
        """
        Close the gripper using the supplied position and effort.
        """

        self.send_gripper_goal(
            position=position,
            max_effort=max_effort
        )

    def grasp_and_lift(self, close_position, max_effort, z_offset):
        """Close the fingers, then physically attach and lift the box."""

        self._pending_lift_offset = float(z_offset)
        self.close_gripper(close_position, max_effort)

    def gripper_goal_response_callback(self, future):
        """Process the response after sending a gripper goal."""

        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Failed to send the gripper goal: {error}'
            )
            return

        if not goal_handle.accepted:
            self.get_logger().error(
                'The gripper goal was rejected'
            )
            return

        self.get_logger().info(
            'The gripper goal was accepted'
        )

        # Keep a fallback for a simulated trajectory controller that does not
        # return after the commanded two-second finger trajectory.
        if self._pending_lift_offset is not None:
            self._gripper_completion_timer = self.create_timer(
                3.0,
                self.gripper_completion_timeout_callback
            )

        result_future = goal_handle.get_result_async()

        result_future.add_done_callback(
            self.gripper_result_callback
        )

    def request_pending_lift(self, reason):
        """Request attach/lift once after gripper completion or timeout."""

        if self._pending_lift_offset is None:
            return

        if self._gripper_completion_timer is not None:
            self._gripper_completion_timer.cancel()
            self.destroy_timer(self._gripper_completion_timer)
            self._gripper_completion_timer = None

        z_offset = self._pending_lift_offset
        self._pending_lift_offset = None
        self.get_logger().info(
            f'{reason}; requesting Gazebo attach and Cartesian lift'
        )
        self.attach_and_lift(z_offset)

    def gripper_completion_timeout_callback(self):
        """Continue a simulated grasp if the gripper action never finishes."""

        self.get_logger().warning(
            'Gripper action did not return a result within 3 seconds'
        )
        self.request_pending_lift('Gripper close timeout reached')

    def gripper_feedback_callback(self, feedback_msg):
        """Receive feedback while the gripper is moving."""

        feedback = feedback_msg.feedback
        actual_positions = list(feedback.actual.positions)
        desired_positions = list(feedback.desired.positions)

        self.get_logger().info(
            f'Gripper feedback: '
            f'desired_positions={desired_positions}, '
            f'actual_positions={actual_positions}, '
            f'measured_effort={self._latest_gripper_effort}, '
            f'peak_measured_effort={self._peak_gripper_effort:.2f}'
        )

    def gripper_result_callback(self, future):
        """Process the final gripper result."""

        try:
            wrapped_result = future.result()
            result = wrapped_result.result
            status = wrapped_result.status
        except Exception as error:
            self.get_logger().error(
                f'Failed to receive the gripper result: {error}'
            )
            return

        self.get_logger().info(
            f'Gripper action completed: '
            f'error_code={result.error_code}, '
            f'error_string="{result.error_string}", '
            f'measured_effort={self._latest_gripper_effort}, '
            f'peak_measured_effort={self._peak_gripper_effort:.2f}, '
            f'status={status}'
        )

        if self._pick_sequence is not None:
            if result.error_code != 0:
                self.abort_pick_sequence(
                    f'gripper stage {self._pick_sequence["stage"]} failed: '
                    f'{result.error_string}'
                )
                return

            stage = self._pick_sequence['stage']
            if stage == 'opening':
                self._pick_sequence['stage'] = 'pregrasp'
                self.get_logger().info(
                    'Bottle grasp stage 1/6: automatically planning to the '
                    'side pre-grasp pose'
                )
                qx, qy, qz, qw = self._pick_sequence['grasp_quaternion']
                self.send_pose_goal(
                    self._pick_sequence['approach_x'],
                    self._pick_sequence['approach_y'],
                    self._pick_sequence['grasp_z'],
                    qx, qy, qz, qw)
                return

            if stage == 'closing':
                self._pick_sequence['stage'] = 'lifting'
                self.get_logger().info(
                    'Bottle grasp stage 4/6: attaching and lifting the bottle'
                )
                self.attach_and_lift(
                    self._pick_sequence['lift_offset'],
                    self._pick_sequence['object_x'],
                    self._pick_sequence['object_y'],
                    self._pick_sequence['object_z'])
                return

            if stage == 'releasing':
                self.get_logger().info(
                    'Pick-and-move sequence completed successfully; '
                    'the object was released at the target'
                )
                self._pick_sequence = None
                rclpy.shutdown()
                return

        if self._pending_lift_offset is not None:
            self.request_pending_lift('Gripper action completed')
            return

        rclpy.shutdown()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            'Side-grasp a bottle at an XYZ position expressed in base_link.'
        ),
    )
    parser.add_argument(
        '--object-x', type=float, default=FIXED_BOTTLE_X,
        help='Object reference X in base_link (metres).')
    parser.add_argument(
        '--object-y', type=float, default=FIXED_BOTTLE_Y,
        help='Object reference Y in base_link (metres).')
    parser.add_argument(
        '--object-z', type=float, default=FIXED_BOTTLE_Z,
        help=(
            'Object reference Z in base_link (metres); the simulated bottle '
            'uses its model origin near the bottom.'
        ))
    parser.add_argument(
        '--approach-clearance', type=float, default=FIXED_APPROACH_CLEARANCE)
    parser.add_argument(
        '--tcp-grasp-offset', type=float, default=FIXED_TCP_GRASP_OFFSET,
        help=(
            'Vertical offset from object Z to the desired TCP grasp height; '
            'use 0 if object Z already denotes the grasp centre.'
        ))
    parser.add_argument(
        '--grasp-depth-offset', type=float, default=FIXED_GRASP_DEPTH_OFFSET)
    parser.add_argument('--lift-offset', type=float, default=FIXED_LIFT_OFFSET)
    parser.add_argument('--open-position', type=float, default=FIXED_OPEN_POSITION)
    parser.add_argument('--close-position', type=float, default=FIXED_CLOSE_POSITION)
    parser.add_argument('--max-effort', type=float, default=FIXED_MAX_EFFORT)
    parser.add_argument(
        '--pour-angle-deg', type=float, default=FIXED_POUR_WRIST_ANGLE_DEG,
        help=(
            'Signed wrist_3 rotation after transport; change the sign to '
            'reverse the pouring direction.'
        ))
    parser.add_argument(
        '--transport-dx', type=float, default=None,
        help='Post-lift base_link X offset; default chooses a small safe value.')
    parser.add_argument(
        '--transport-dy', type=float, default=None,
        help='Post-lift base_link Y offset; default chooses a small safe value.')

    parsed = parser.parse_args(remove_ros_args(args=argv)[1:])
    if math.hypot(parsed.object_x, parsed.object_y) < 1e-6:
        parser.error('object X/Y cannot both be zero in base_link')
    if parsed.object_z < 0.0:
        parser.error('--object-z must be non-negative')
    if parsed.approach_clearance <= 0.0:
        parser.error('--approach-clearance must be positive')
    if parsed.tcp_grasp_offset < 0.0:
        parser.error('--tcp-grasp-offset must be non-negative')
    if parsed.grasp_depth_offset < 0.0:
        parser.error('--grasp-depth-offset must be non-negative')
    if parsed.lift_offset <= 0.0:
        parser.error('--lift-offset must be positive')
    if not 1.0 <= abs(parsed.pour_angle_deg) <= 150.0:
        parser.error('--pour-angle-deg magnitude must be between 1 and 150')
    if not 0.0 <= parsed.close_position <= parsed.open_position <= 0.06:
        parser.error(
            'gripper positions must satisfy '
            '0 <= close-position <= open-position <= 0.06')
    if (
        parsed.transport_dx is not None
        and parsed.transport_dy is not None
        and max(abs(parsed.transport_dx), abs(parsed.transport_dy)) < 1e-6
    ):
        parser.error('transport DX/DY cannot both be zero')
    return parsed


def main(args=None):
    argv = sys.argv if args is None else args
    cli_args = parse_args(argv)
    rclpy.init(args=argv)

    action_client = UR3ActionClient()

    sequence_started = action_client.grasp_bottle_at(
        cli_args.object_x,
        cli_args.object_y,
        cli_args.object_z,
        approach_clearance=cli_args.approach_clearance,
        tcp_grasp_offset=cli_args.tcp_grasp_offset,
        grasp_depth_offset=cli_args.grasp_depth_offset,
        lift_offset=cli_args.lift_offset,
        open_position=cli_args.open_position,
        close_position=cli_args.close_position,
        max_effort=cli_args.max_effort,
        pour_angle_deg=cli_args.pour_angle_deg,
        transport_dx=cli_args.transport_dx,
        transport_dy=cli_args.transport_dy,
    )

    if not sequence_started:
        action_client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        rclpy.spin(action_client)

    except KeyboardInterrupt:
        action_client.get_logger().info(
            'The program was stopped by the user'
        )

    finally:
        action_client.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
