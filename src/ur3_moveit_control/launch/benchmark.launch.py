import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def load_yaml(package_name, file_path):
    try:
        package_path = get_package_share_directory(package_name)
        absolute_path = os.path.join(package_path, file_path)
        with open(absolute_path, "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def launch_setup(context, *args, **kwargs):
    ur_type = LaunchConfiguration("ur_type")

    # 1. Load Robot Model (URDF / Xacro)
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare("ur3_moveit_control"), "urdf", "ur3_with_susgrip.urdf.xacro"]), " ",
        "ur_type:=", ur_type, " ",
        "sim_ignition:=true", " ",
        "name:=", "ur",
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # 2. Load Semantic Model (SRDF)
    robot_description_semantic_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]), " ",
        PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "srdf", "ur3_with_susgrip.srdf.xacro"]), " ",
        "name:=", "ur",
    ])
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(robot_description_semantic_content, value_type=str)
    }

    # 3. Load Kinematics and Joint Limits
    kinematics_yaml = {"robot_description_kinematics": load_yaml("ur3_moveit_control", "config/kinematics.yaml")}
    joint_limits_yaml = {"robot_description_planning": load_yaml("ur3_moveit_control", "config/joint_limits.yaml")}

    # 4. Load planning pipelines and merge OMPL configs
    planning_pipelines_yaml = load_yaml("ur3_moveit_control", "config/planning_pipelines.yaml")
    ompl_planning_yaml = load_yaml("ur3_moveit_control", "config/ompl_planning.yaml")
    if planning_pipelines_yaml and ompl_planning_yaml:
        planning_pipelines_yaml["planning_pipelines"]["ompl"].update(ompl_planning_yaml)

    # 5. Đường dẫn tuyệt đối tới file benchmark_ur3.yaml

    benchmark_yaml_path = os.path.join(
        get_package_share_directory("ur3_moveit_control"),
        "config",
        "benchmark_ur3.yaml",
    )

    # Đảm bảo thư mục lưu kết quả tồn tại
    os.makedirs("/tmp/moveit_benchmarks/", exist_ok=True)

        # 6. Ép MoveIt sử dụng OMPL Planner Plugin (Tránh rơi vào CHOMPPlanner)
    ompl_plugin_override = {
        "ompl.planning_plugin": "ompl_interface/OMPLPlanner",
        "planning_plugin": "ompl_interface/OMPLPlanner",
    }

    # 7. MoveIt Benchmark Node
    benchmark_node = Node(
        package="moveit_ros_benchmarks",
        executable="moveit_run_benchmark",
        name="moveit_run_benchmark",
        output="screen",
        arguments=["--benchmark-options", benchmark_yaml_path], # Đọc trực tiếp YAML
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            joint_limits_yaml,
            planning_pipelines_yaml,
            ompl_plugin_override,
            benchmark_yaml_path,
        ],
    )

    return [benchmark_node]

def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur3e", description="Type of UR robot"),
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
