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


def flatten_dict(d, parent=""):
    """Flatten nested dict thành dot-notation keys mà ROS2 param server hiểu."""
    items = {}
    for k, v in d.items():
        key = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        else:
            items[key] = v
    return items


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

    # 4. Load OMPL planning pipeline config
    planning_pipelines_yaml = load_yaml("ur3_moveit_control", "config/planning_pipelines.yaml")
    ompl_planning_yaml = load_yaml("ur3_moveit_control", "config/ompl_planning.yaml")

    ompl_section = {}
    if planning_pipelines_yaml and "planning_pipelines" in planning_pipelines_yaml:
        # Trích xuất riêng nhánh ompl
        ompl_section = planning_pipelines_yaml["planning_pipelines"].get("ompl", {})
        
        # Chuyển list request_adapters thành string cách nhau bởi space (BẮT BUỘC CHO MOVEIT)
        if "request_adapters" in ompl_section and isinstance(ompl_section["request_adapters"], list):
            ompl_section["request_adapters"] = " ".join(ompl_section["request_adapters"])

    # Gộp thêm tham số chi tiết của từng thuật toán (RRT, PRM...) từ ompl_planning.yaml
    if ompl_planning_yaml:
        ompl_section.update(ompl_planning_yaml)

    # Flatten thành dạng "ompl.planning_plugin", "ompl.request_adapters"...
    ompl_pipeline_params = flatten_dict({"ompl": ompl_section})

    # Cấu hình danh sách pipeline cho MoveIt
    pipeline_names_config = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
    }

    # 5. Load benchmark config và flatten thành dot-notation params cho ROS2
    os.makedirs("/tmp/moveit_benchmarks/", exist_ok=True)
    benchmark_yaml = load_yaml("ur3_moveit_control", "config/benchmark_ur3.yaml")
    BENCHMARK_PARAM_SECTIONS = {"parameters", "planning_pipelines", "warehouse"}
    benchmark_params = {}
    if benchmark_yaml and "benchmark_config" in benchmark_yaml:
        filtered = {
            k: v for k, v in benchmark_yaml["benchmark_config"].items()
            if k in BENCHMARK_PARAM_SECTIONS
        }
        benchmark_params = flatten_dict({"benchmark_config": filtered})

    # 6. Warehouse plugin – dùng SQLite
    warehouse_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": "/home/tu/.ros/warehouse_ros.sqlite",
        "warehouse_port": 0,
    }

    # 7. MoveIt Benchmark Node
    benchmark_node = Node(
        package="moveit_ros_benchmarks",
        executable="moveit_run_benchmark",
        name="moveit_run_benchmark",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            joint_limits_yaml,
            ompl_pipeline_params,       # {"ompl": {"planning_plugin": "ompl_interface/OMPLPlanner", ...}}
            pipeline_names_config,      # {"planning_pipelines": ["ompl"]}
            warehouse_config,
            benchmark_params,
        ],
    )

    return [benchmark_node]


def generate_launch_description():
    declared_arguments = [
        DeclareLaunchArgument("ur_type", default_value="ur3e", description="Type of UR robot"),
    ]
    return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)])
