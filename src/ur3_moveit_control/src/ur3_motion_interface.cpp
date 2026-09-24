#include "ur3_moveit_control/ur3_motion_interface.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <shape_msgs/msg/solid_primitive.hpp>

namespace
{

constexpr double kTwoPi = 6.28318530717958647692;

struct WaypointSeed
{
  std::vector<double> joints;
  double pose_distance{0.0};
  std::string source;
};

struct IkCandidate
{
  std::vector<double> joints;
  double goal_score{0.0};
  std::string source;
};

std::vector<std::string> splitCsvLine(const std::string & line)
{
  std::vector<std::string> cells;
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ','))
  {
    if (!cell.empty() && cell.back() == '\r')
    {
      cell.pop_back();
    }
    cells.push_back(cell);
  }
  return cells;
}

double quaternionDistance(
  const geometry_msgs::msg::Quaternion & lhs,
  double qx, double qy, double qz, double qw)
{
  const double lhs_norm = std::sqrt(
    lhs.x * lhs.x + lhs.y * lhs.y + lhs.z * lhs.z + lhs.w * lhs.w);
  const double rhs_norm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (lhs_norm < 1e-9 || rhs_norm < 1e-9)
  {
    return std::numeric_limits<double>::infinity();
  }

  double dot = std::abs(
    (lhs.x * qx + lhs.y * qy + lhs.z * qz + lhs.w * qw) /
    (lhs_norm * rhs_norm));
  dot = std::max(0.0, std::min(1.0, dot));
  return 2.0 * std::acos(dot);
}

std::vector<WaypointSeed> loadWaypointSeeds(
  const std::vector<std::string> & files,
  const geometry_msgs::msg::Pose & target_pose,
  const std::vector<std::string> & joint_names,
  double max_position_distance,
  double max_orientation_distance)
{
  std::vector<WaypointSeed> seeds;
  const std::vector<std::string> pose_fields = {
    "pregrasp_x", "pregrasp_y", "pregrasp_z", "qx", "qy", "qz", "qw"};

  for (const auto & file_name : files)
  {
    std::ifstream csv_file(file_name);
    if (!csv_file.is_open())
    {
      continue;
    }

    std::string line;
    if (!std::getline(csv_file, line))
    {
      continue;
    }

    const auto header = splitCsvLine(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index)
    {
      columns[header[index]] = index;
    }

    bool required_columns_present = true;
    for (const auto & field : pose_fields)
    {
      required_columns_present = required_columns_present && columns.count(field) > 0;
    }
    for (const auto & joint_name : joint_names)
    {
      required_columns_present = required_columns_present && columns.count(joint_name) > 0;
    }
    if (!required_columns_present)
    {
      continue;
    }

    while (std::getline(csv_file, line))
    {
      const auto cells = splitCsvLine(line);
      const auto value = [&cells, &columns](const std::string & name) {
          const auto index = columns.at(name);
          if (index >= cells.size())
          {
            throw std::out_of_range("CSV row is shorter than its header");
          }
          return std::stod(cells[index]);
        };

      try
      {
        const double x = value("pregrasp_x");
        const double y = value("pregrasp_y");
        const double z = value("pregrasp_z");
        const double dx = x - target_pose.position.x;
        const double dy = y - target_pose.position.y;
        const double dz = z - target_pose.position.z;
        const double position_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double orientation_distance = quaternionDistance(
          target_pose.orientation,
          value("qx"), value("qy"), value("qz"), value("qw"));

        if (position_distance > max_position_distance ||
            orientation_distance > max_orientation_distance)
        {
          continue;
        }

        WaypointSeed seed;
        seed.pose_distance = position_distance + 0.10 * orientation_distance;
        seed.source = file_name;
        for (const auto & joint_name : joint_names)
        {
          seed.joints.push_back(value(joint_name));
        }
        seeds.push_back(std::move(seed));
      }
      catch (const std::exception &)
      {
        // Ignore incomplete rows so one interrupted CSV write cannot disable
        // all previously proven waypoints.
      }
    }
  }

  std::sort(
    seeds.begin(), seeds.end(),
    [](const WaypointSeed & lhs, const WaypointSeed & rhs) {
      return lhs.pose_distance < rhs.pose_distance;
    });
  return seeds;
}

void normalizeNearReference(
  std::vector<double> & joints,
  const std::vector<double> & reference,
  const std::vector<std::string> & joint_names,
  const moveit::core::RobotModelConstPtr & robot_model)
{
  const std::size_t count = std::min(
    joints.size(), std::min(reference.size(), joint_names.size()));
  for (std::size_t index = 0; index < count; ++index)
  {
    const auto & bounds = robot_model->getVariableBounds(joint_names[index]);
    double best_value = joints[index];
    double best_distance = std::abs(best_value - reference[index]);

    for (int turn = -2; turn <= 2; ++turn)
    {
      const double equivalent = joints[index] + static_cast<double>(turn) * kTwoPi;
      if (bounds.position_bounded_ &&
          (equivalent < bounds.min_position_ || equivalent > bounds.max_position_))
      {
        continue;
      }
      const double distance = std::abs(equivalent - reference[index]);
      if (distance < best_distance)
      {
        best_value = equivalent;
        best_distance = distance;
      }
    }
    joints[index] = best_value;
  }
}

double scoreJointGoal(
  const std::vector<double> & joints,
  const std::vector<double> & current,
  const std::vector<double> & weights,
  const std::vector<std::string> & joint_names,
  const moveit::core::RobotModelConstPtr & robot_model)
{
  double score = 0.0;
  const std::size_t count = std::min(joints.size(), current.size());
  for (std::size_t index = 0; index < count; ++index)
  {
    const double weight = index < weights.size() ? weights[index] : 1.0;
    score += weight * std::abs(joints[index] - current[index]);

    if (index < joint_names.size())
    {
      const auto & bounds = robot_model->getVariableBounds(joint_names[index]);
      if (bounds.position_bounded_)
      {
        const double margin = std::min(
          joints[index] - bounds.min_position_,
          bounds.max_position_ - joints[index]);
        if (margin < 0.35)
        {
          score += weight * 4.0 * (0.35 - margin);
        }
      }
    }
  }
  return score;
}

double scorePlan(
  const moveit::planning_interface::MoveGroupInterface::Plan & plan,
  const std::vector<std::string> & group_joint_names,
  const std::vector<double> & group_weights)
{
  const auto & trajectory = plan.trajectory_.joint_trajectory;
  if (trajectory.points.empty())
  {
    return std::numeric_limits<double>::infinity();
  }
  if (trajectory.points.size() == 1)
  {
    return 0.0;
  }

  std::unordered_map<std::string, double> weights;
  for (std::size_t index = 0; index < group_joint_names.size(); ++index)
  {
    weights[group_joint_names[index]] =
      index < group_weights.size() ? group_weights[index] : 1.0;
  }

  double score = 0.0;
  for (std::size_t point = 1; point < trajectory.points.size(); ++point)
  {
    const auto & previous = trajectory.points[point - 1].positions;
    const auto & current = trajectory.points[point].positions;
    const std::size_t count = std::min(
      trajectory.joint_names.size(), std::min(previous.size(), current.size()));
    for (std::size_t joint = 0; joint < count; ++joint)
    {
      const auto weight_it = weights.find(trajectory.joint_names[joint]);
      const double weight = weight_it == weights.end() ? 1.0 : weight_it->second;
      const double step = std::abs(current[joint] - previous[joint]);
      score += weight * step;
      if (step > 3.14159265358979323846)
      {
        score += 100.0 * step;
      }
    }
  }
  return score;
}

bool hasUnsafeCartesianJointStep(
  const moveit_msgs::msg::RobotTrajectory & trajectory,
  double maximum_step,
  double & observed_maximum)
{
  observed_maximum = 0.0;
  const auto & points = trajectory.joint_trajectory.points;
  for (std::size_t point = 1; point < points.size(); ++point)
  {
    const auto & previous = points[point - 1].positions;
    const auto & current = points[point].positions;
    const std::size_t count = std::min(previous.size(), current.size());
    for (std::size_t joint = 0; joint < count; ++joint)
    {
      observed_maximum = std::max(
        observed_maximum, std::abs(current[joint] - previous[joint]));
    }
  }
  return observed_maximum > maximum_step;
}

}  // namespace

namespace ur3_moveit_control
{

  UR3MotionInterface::UR3MotionInterface(
      const rclcpp::Node::SharedPtr &node,
      const std::string &planning_group)
      : node_(node),
        move_group_(node_, planning_group)
  {
    const auto parameter = [this](const std::string & name, const auto & default_value) {
        using ParameterType = std::decay_t<decltype(default_value)>;
        if (!node_->has_parameter(name))
        {
          node_->declare_parameter<ParameterType>(name, default_value);
        }
        return node_->get_parameter(name).get_value<ParameterType>();
      };

    successful_waypoint_files_ = parameter(
      "successful_waypoint_files",
      std::vector<std::string>{
        "data/successful_waypoints.csv"});
    ik_joint_weights_ = parameter(
      "ik_joint_weights", std::vector<double>{1.2, 1.0, 1.0, 3.0, 2.5, 2.0});
    ik_random_seed_attempts_ = parameter("ik_random_seed_attempts", 12);
    ik_max_planning_candidates_ = parameter("ik_max_planning_candidates", 8);
    ik_max_successful_plans_ = parameter("ik_max_successful_plans", 3);
    waypoint_seed_max_position_distance_ = parameter(
      "waypoint_seed_max_position_distance", 0.20);
    waypoint_seed_max_orientation_distance_ = parameter(
      "waypoint_seed_max_orientation_distance", 0.35);
    cartesian_jump_threshold_ = parameter("cartesian_jump_threshold", 1.5);
    cartesian_max_joint_step_ = parameter("cartesian_max_joint_step", 0.20);
    attach_confirmation_timeout_ = parameter("attach_confirmation_timeout", 3.0);
    attach_request_attempts_ = std::max(
      1, static_cast<int>(parameter("attach_request_attempts", 3)));
    maximum_attach_tcp_distance_ = parameter("maximum_attach_tcp_distance", 0.06);
    maximum_attach_tcp_z_error_ = parameter("maximum_attach_tcp_z_error", 0.04);
    maximum_gripper_position_for_attach_ = parameter(
      "maximum_gripper_position_for_attach", 0.051);
    minimum_gripper_effort_for_attach_ = parameter(
      "minimum_gripper_effort_for_attach", 0.0001);

    gazebo_attach_pub_ = node_->create_publisher<std_msgs::msg::Empty>(
      "/pick_box/attach", rclcpp::QoS(1).reliable());
    gazebo_detach_pub_ = node_->create_publisher<std_msgs::msg::Empty>(
      "/pick_box/detach", rclcpp::QoS(1).reliable());
    attachment_state_sub_ = node_->create_subscription<std_msgs::msg::String>(
      "/pick_box/attachment_state",
      rclcpp::QoS(10).reliable(),
      std::bind(
        &UR3MotionInterface::attachmentStateCallback, this, std::placeholders::_1));
    joint_state_sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states",
      rclcpp::SensorDataQoS(),
      std::bind(&UR3MotionInterface::jointStateCallback, this, std::placeholders::_1));

    move_group_.setPlanningTime(15.0);
    move_group_.setNumPlanningAttempts(20);

    // Use the gripper grasp center as the Cartesian pose target whenever the
    // gripper model is present. Keep tool0 as the fallback for the arm-only
    // robot description.
    if (move_group_.getRobotModel()->getLinkModel("gripper_tcp") != nullptr)
    {
      const bool tcp_set = move_group_.setEndEffectorLink("gripper_tcp");
      if (!tcp_set)
      {
        RCLCPP_WARN(
          node_->get_logger(),
          "gripper_tcp exists but could not be selected; using %s instead.",
          move_group_.getEndEffectorLink().c_str());
      }
    }

    double max_velocity_scaling = node_->get_parameter("max_velocity_scaling_factor").as_double();
    double max_acceleration_scaling = node_->get_parameter("max_acceleration_scaling_factor").as_double();
    std::string planner_id = node_->get_parameter("planner_id").as_string();
    double goal_pos_tol = node_->get_parameter("goal_position_tolerance").as_double();
    double goal_ori_tol = node_->get_parameter("goal_orientation_tolerance").as_double();
    double goal_joint_tol = node_->get_parameter("goal_joint_tolerance").as_double();

    move_group_.setMaxVelocityScalingFactor(max_velocity_scaling);
    move_group_.setMaxAccelerationScalingFactor(max_acceleration_scaling);

    // Set Planning Pipeline and Planner ID
    // move_group_.setPlanningPipelineId("ompl");
    move_group_.setPlannerId(planner_id);

    // Set Goal Tolerances
    move_group_.setGoalPositionTolerance(goal_pos_tol);
    move_group_.setGoalOrientationTolerance(goal_ori_tol);
    move_group_.setGoalJointTolerance(goal_joint_tol);

    RCLCPP_INFO(
        node_->get_logger(),
        "Planner ID: %s",
        planner_id.c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Planning group: %s",
        planning_group.c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Planning frame: %s",
        move_group_.getPlanningFrame().c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "End-effector link: %s",
        move_group_.getEndEffectorLink().c_str());

    RCLCPP_INFO(
        node_->get_logger(),
        "Max velocity scaling factor: %.2f",
        max_velocity_scaling);

    RCLCPP_INFO(
        node_->get_logger(),
        "Max acceleration scaling factor: %.2f",
        max_acceleration_scaling);

    RCLCPP_INFO(
      node_->get_logger(),
      "Smooth planning: %d random IK seeds, at most %d candidates / %d "
      "successful plans, Cartesian jump threshold %.2f, max joint step %.3f rad.",
      ik_random_seed_attempts_, ik_max_planning_candidates_,
      ik_max_successful_plans_, cartesian_jump_threshold_,
      cartesian_max_joint_step_);
  }

  void UR3MotionInterface::attachmentStateCallback(
    const std_msgs::msg::String::SharedPtr msg)
  {
    bool attached = false;
    if (msg->data == "attached")
    {
      attached = true;
    }
    else if (msg->data != "detached")
    {
      RCLCPP_WARN(
        node_->get_logger(),
        "Ignoring unknown Gazebo attachment state '%s'; expected 'attached' or 'detached'.",
        msg->data.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(attachment_state_mutex_);
      attachment_state_ = attached;
      attachment_state_received_ = true;
    }
    RCLCPP_INFO(
      node_->get_logger(), "Gazebo attachment state: %s", msg->data.c_str());
    attachment_state_cv_.notify_all();
  }

  void UR3MotionInterface::jointStateCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const auto position_it = std::find(
      msg->name.begin(), msg->name.end(), "gripper_joint");
    if (position_it == msg->name.end())
    {
      return;
    }

    const std::size_t index = static_cast<std::size_t>(
      std::distance(msg->name.begin(), position_it));
    if (index >= msg->position.size())
    {
      return;
    }

    const double new_position = msg->position[index];
    const double new_effort = index < msg->effort.size() ? msg->effort[index] : 0.0;

    std::lock_guard<std::mutex> lock(gripper_state_mutex_);
    if (gripper_state_received_ &&
        gripper_position_ > maximum_gripper_position_for_attach_ &&
        new_position <= maximum_gripper_position_for_attach_)
    {
      // Crossing from open to closed marks a new grasp attempt.
      gripper_close_peak_effort_ = 0.0;
    }
    if (new_position <= maximum_gripper_position_for_attach_)
    {
      gripper_close_peak_effort_ = std::max(
        gripper_close_peak_effort_, std::abs(new_effort));
    }
    gripper_position_ = new_position;
    gripper_effort_ = new_effort;
    gripper_state_received_ = true;
  }

  bool UR3MotionInterface::waitForAttachmentState(
    bool expected_state, double timeout_seconds)
  {
    std::unique_lock<std::mutex> lock(attachment_state_mutex_);
    return attachment_state_cv_.wait_for(
      lock,
      std::chrono::duration<double>(timeout_seconds),
      [this, expected_state]() {
        return attachment_state_received_ && attachment_state_ == expected_state;
      });
  }

  bool UR3MotionInterface::moveToJointGoal(
      const std::vector<double> &joint_goal)
  {
    if (joint_goal.size() != 6)
    {
      RCLCPP_ERROR(
          node_->get_logger(),
          "UR3 joint goal must contain exactly 6 joint values.");
      return false;
    }

    move_group_.setJointValueTarget(joint_goal);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const bool planning_success =
        static_cast<bool>(move_group_.plan(plan));

    if (!planning_success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Joint planning failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Joint planning succeeded. Executing...");

    const auto execution_result = move_group_.execute(plan);

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Joint execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Joint execution succeeded.");
    return true;
  }

  bool UR3MotionInterface::moveToPoseGoal(
      const geometry_msgs::msg::Pose &target_pose,
      const std::string &end_effector_link)
  {
    last_pose_error_message_.clear();

    std::string target_link = end_effector_link;
    if (!end_effector_link.empty())
    {
      move_group_.setEndEffectorLink(end_effector_link);
    }
    else
    {
      target_link = move_group_.getEndEffectorLink();
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    moveit::planning_interface::MoveGroupInterface::Plan best_plan;
    bool planning_success = false;
    int ik_solutions_found = 0;

    // Generate explicit joint-space IK goals, rank them before planning, then
    // compare several successful trajectories. This prevents the first random
    // elbow/wrist branch from winning merely because OMPL happened to find it
    // first.
    const auto current_state = move_group_.getCurrentState(2.0);
    const auto *joint_model_group =
      move_group_.getRobotModel()->getJointModelGroup(move_group_.getName());

    if (current_state && joint_model_group)
    {
      const auto joint_names = joint_model_group->getVariableNames();
      std::vector<double> current_joints;
      current_state->copyJointGroupPositions(joint_model_group, current_joints);
      std::vector<IkCandidate> candidates;

      const auto add_ik_candidate = [
        this, &candidates, &current_state, joint_model_group, &target_pose,
        &target_link, &current_joints, &joint_names, &ik_solutions_found](
        const std::string & source,
        const std::vector<double> & seed_joints,
        bool random_seed)
      {
        moveit::core::RobotState ik_state(*current_state);
        if (random_seed)
        {
          ik_state.setToRandomPositions(joint_model_group);
        }
        else if (!seed_joints.empty() && seed_joints.size() == joint_names.size())
        {
          auto normalized_seed = seed_joints;
          normalizeNearReference(
            normalized_seed, current_joints, joint_names,
            move_group_.getRobotModel());
          ik_state.setJointGroupPositions(joint_model_group, normalized_seed);
          ik_state.update();
        }

        if (!ik_state.setFromIK(
            joint_model_group, target_pose, target_link, 0.25))
        {
          return;
        }

        ++ik_solutions_found;
        std::vector<double> joint_values;
        ik_state.copyJointGroupPositions(joint_model_group, joint_values);
        normalizeNearReference(
          joint_values, current_joints, joint_names,
          move_group_.getRobotModel());

        for (const auto & existing : candidates)
        {
          double maximum_difference = 0.0;
          for (std::size_t index = 0; index < joint_values.size(); ++index)
          {
            maximum_difference = std::max(
              maximum_difference,
              std::abs(joint_values[index] - existing.joints[index]));
          }
          if (maximum_difference < 1e-3)
          {
            return;
          }
        }

        IkCandidate candidate;
        candidate.joints = std::move(joint_values);
        candidate.goal_score = scoreJointGoal(
          candidate.joints, current_joints, ik_joint_weights_, joint_names,
          move_group_.getRobotModel());
        candidate.source = source;
        candidates.push_back(std::move(candidate));
      };

      // The current state is the best generic seed because KDL tends to stay
      // on the nearby IK branch.
      add_ik_candidate("current-state seed", {}, false);

      // Successful CSV rows are proven pre-grasp branches. Use the closest
      // rows as IK seeds, but still solve IK for the exact requested pose.
      const auto waypoint_seeds = loadWaypointSeeds(
        successful_waypoint_files_, target_pose, joint_names,
        waypoint_seed_max_position_distance_,
        waypoint_seed_max_orientation_distance_);
      const std::size_t waypoint_limit = std::min<std::size_t>(6, waypoint_seeds.size());
      for (std::size_t index = 0; index < waypoint_limit; ++index)
      {
        add_ik_candidate(
          "saved waypoint " + waypoint_seeds[index].source,
          waypoint_seeds[index].joints,
          false);
      }

      for (int attempt = 0; attempt < std::max(0, ik_random_seed_attempts_); ++attempt)
      {
        add_ik_candidate(
          "random seed " + std::to_string(attempt + 1), {}, true);
      }

      std::sort(
        candidates.begin(), candidates.end(),
        [](const IkCandidate & lhs, const IkCandidate & rhs) {
          return lhs.goal_score < rhs.goal_score;
        });

      RCLCPP_INFO(
        node_->get_logger(),
        "Found %d IK solutions (%zu unique); %zu matching saved waypoint "
        "seeds were available.",
        ik_solutions_found, candidates.size(), waypoint_seeds.size());

      const int candidates_to_plan = std::min<int>(
        std::max(1, ik_max_planning_candidates_),
        static_cast<int>(candidates.size()));
      const int successful_plan_limit = std::max(1, ik_max_successful_plans_);
      int successful_plans = 0;
      double best_plan_score = std::numeric_limits<double>::infinity();

      for (int index = 0; index < candidates_to_plan; ++index)
      {
        const auto & candidate = candidates[static_cast<std::size_t>(index)];

        move_group_.setStartStateToCurrentState();
        move_group_.clearPoseTargets();
        if (!move_group_.setJointValueTarget(candidate.joints))
        {
          continue;
        }

        RCLCPP_INFO(
          node_->get_logger(),
          "Planning ranked IK candidate %d/%d: score=%.3f, source=%s",
          index + 1, candidates_to_plan, candidate.goal_score,
          candidate.source.c_str());
        if (static_cast<bool>(move_group_.plan(plan)))
        {
          ++successful_plans;
          const double trajectory_score =
            scorePlan(plan, joint_names, ik_joint_weights_) +
            0.15 * candidate.goal_score;
          RCLCPP_INFO(
            node_->get_logger(),
            "Candidate %d planned successfully with weighted path score %.3f.",
            index + 1, trajectory_score);
          if (trajectory_score < best_plan_score)
          {
            best_plan_score = trajectory_score;
            best_plan = plan;
            planning_success = true;
          }
          if (successful_plans >= successful_plan_limit)
          {
            break;
          }
        }
      }

      if (planning_success)
      {
        plan = best_plan;
        RCLCPP_INFO(
          node_->get_logger(),
          "Selected the smoothest of %d successful ranked plans "
          "(score %.3f).",
          successful_plans, best_plan_score);
      }
    }

    // Final fallback lets MoveIt sample the pose constraint itself in case its
    // planning-scene-aware goal sampler can find a branch missed above.
    if (!planning_success)
    {
      RCLCPP_WARN(
        node_->get_logger(),
        "No ranked plan found from %d IK solutions; trying sampled pose goal.",
        ik_solutions_found);
      move_group_.setStartStateToCurrentState();
      move_group_.clearPoseTargets();
      if (move_group_.setPoseTarget(target_pose, target_link))
      {
        planning_success = static_cast<bool>(move_group_.plan(plan));
      }
    }

    if (!planning_success)
    {
      last_pose_error_message_ =
        "Pose planning failed: no collision-free IK trajectory was found";
      RCLCPP_ERROR(
        node_->get_logger(),
        "Pose planning failed after testing %d explicit IK candidates and "
        "the sampled pose-goal fallback.",
        ik_solutions_found);
      return false;
    }

    move_group_.clearPoseTargets();

    RCLCPP_INFO(node_->get_logger(), "Pose planning succeeded. Executing...");

    const auto execution_result = move_group_.execute(plan);

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      last_pose_error_message_ =
        "Pose planning succeeded, but trajectory execution failed; check "
        "controller path tolerance / tracking";
      RCLCPP_ERROR(
        node_->get_logger(),
        "Pose execution failed after successful planning. The trajectory "
        "controller rejected or could not track the path.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Pose execution succeeded.");
    return true;
  }

  const std::string & UR3MotionInterface::lastPoseErrorMessage() const
  {
    return last_pose_error_message_;
  }

  bool UR3MotionInterface::moveHome()
  {
    const bool target_set = move_group_.setNamedTarget("home");
    if (!target_set)
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to set named target: home");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    const bool planning_success =
        static_cast<bool>(move_group_.plan(plan));

    if (!planning_success)
    {
      RCLCPP_ERROR(node_->get_logger(), "Home planning failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Home planning succeeded. Executing...");

    const auto execution_result = move_group_.execute(plan);

    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Home execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Home execution succeeded.");
    return true;
  }

  bool UR3MotionInterface::attachPickBox(
    double object_x, double object_y, double object_z)
  {
    const std::string object_id = "pick_box";
    const std::string attach_link = "gripper_tcp";
    const std::vector<std::string> touch_links = {
      "gripper_tcp",
      "gripper_body",
      "finger_left",
      "finger_right"
    };

    // Do not let the fixed-joint plugin hide a missed grasp. The TCP must be
    // next to the bottle body and the gripper must actually have completed a
    // closing movement before an attach request is allowed.
    const auto tcp_pose = move_group_.getCurrentPose(attach_link);
    const double tcp_dx = tcp_pose.pose.position.x - object_x;
    const double tcp_dy = tcp_pose.pose.position.y - object_y;
    const double tcp_distance = std::sqrt(tcp_dx * tcp_dx + tcp_dy * tcp_dy);
    const double expected_grasp_z = object_z + 0.06;
    const double tcp_z_error = std::abs(
      tcp_pose.pose.position.z - expected_grasp_z);

    if (tcp_distance > maximum_attach_tcp_distance_ ||
        tcp_z_error > maximum_attach_tcp_z_error_)
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Refusing attach: gripper_tcp is not at the bottle grasp region "
        "(XY distance %.3f/%.3f m, Z error %.3f/%.3f m).",
        tcp_distance, maximum_attach_tcp_distance_,
        tcp_z_error, maximum_attach_tcp_z_error_);
      return false;
    }

    double gripper_position = 0.0;
    double gripper_effort = 0.0;
    double gripper_peak_effort = 0.0;
    {
      std::lock_guard<std::mutex> lock(gripper_state_mutex_);
      if (!gripper_state_received_)
      {
        RCLCPP_ERROR(
          node_->get_logger(),
          "Refusing attach: no gripper_joint state has been received.");
        return false;
      }
      gripper_position = gripper_position_;
      gripper_effort = gripper_effort_;
      gripper_peak_effort = gripper_close_peak_effort_;
    }

    if (gripper_position > maximum_gripper_position_for_attach_)
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Refusing attach: gripper is still open at %.4f m (must be <= %.4f m).",
        gripper_position, maximum_gripper_position_for_attach_);
      return false;
    }
    if (gripper_peak_effort < minimum_gripper_effort_for_attach_)
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Refusing attach: close effort peak %.6f is below the configured "
        "contact threshold %.6f.",
        gripper_peak_effort, minimum_gripper_effort_for_attach_);
      return false;
    }

    RCLCPP_INFO(
      node_->get_logger(),
      "Attach preconditions passed: TCP distance %.3f m, Z error %.3f m, "
      "gripper position %.4f m, effort %.6f (close peak %.6f).",
      tcp_distance, tcp_z_error, gripper_position,
      gripper_effort, gripper_peak_effort);

    // Keep the box out of the world collision scene during approach so the
    // fingers are allowed to move around it. Add it immediately before
    // attaching, then MoveIt transfers it from the world to the robot as a
    // carried payload with the touch links allowed below.
    addPickBoxObstacle(object_x, object_y, object_z);
    rclcpp::sleep_for(std::chrono::milliseconds(200));

    // First create the physical fixed joint in Gazebo.  Give the ROS-Gazebo
    // bridge and the DetachableJoint system a few simulation iterations to
    // consume the request before the arm starts moving.
    const double attempt_timeout =
      attach_confirmation_timeout_ / static_cast<double>(attach_request_attempts_);
    bool gazebo_attached = false;
    for (int attempt = 1; attempt <= attach_request_attempts_; ++attempt)
    {
      {
        std::lock_guard<std::mutex> lock(attachment_state_mutex_);
        attachment_state_received_ = false;
      }

      RCLCPP_INFO(
        node_->get_logger(),
        "Gazebo attach request %d/%d.", attempt, attach_request_attempts_);
      gazebo_attach_pub_->publish(std_msgs::msg::Empty{});
      if (waitForAttachmentState(true, attempt_timeout))
      {
        gazebo_attached = true;
        break;
      }

      if (attempt < attach_request_attempts_)
      {
        // If Gazebo created the joint but its state message was lost, force a
        // detached state before retrying. This guarantees the next successful
        // request produces a fresh "attached" transition and confirmation.
        RCLCPP_WARN(
          node_->get_logger(),
          "No attachment confirmation after request %d/%d; resetting the "
          "Gazebo joint before retrying.",
          attempt, attach_request_attempts_);
        gazebo_detach_pub_->publish(std_msgs::msg::Empty{});
        rclcpp::sleep_for(std::chrono::milliseconds(200));
      }
    }

    if (!gazebo_attached)
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Gazebo did not confirm attachment_state='attached' after %d requests "
        "within %.1f seconds; the lift is blocked.",
        attach_request_attempts_, attach_confirmation_timeout_);
      return false;
    }

    // Then mirror that state in MoveIt so collision checking treats the box
    // as carried payload and permits contact with the gripper touch links.
    if (!move_group_.attachObject(object_id, attach_link, touch_links))
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Failed to attach %s to %s in MoveIt; rolling back the Gazebo joint.",
        object_id.c_str(), attach_link.c_str());
      gazebo_detach_pub_->publish(std_msgs::msg::Empty{});
      return false;
    }

    RCLCPP_INFO(
      node_->get_logger(),
      "Gazebo confirmed attachment and MoveIt attached %s to %s.",
      object_id.c_str(), attach_link.c_str());

    // Give the planning scene monitor time to receive the attached-object update.
    rclcpp::sleep_for(std::chrono::milliseconds(200));
    return true;
  }

  bool UR3MotionInterface::detachPickBox()
  {
    const std::string object_id = "pick_box";

    // Save the release X/Y before changing the planning-scene attachment.
    // MoveIt has no gravity simulation, so after detach the scene object is
    // explicitly placed back on the known table surface.
    const auto release_pose = move_group_.getCurrentPose("gripper_tcp");

    // Always release the physical Gazebo joint. Previously a MoveIt scene
    // mismatch returned early and prevented this message from being sent.
    {
      std::lock_guard<std::mutex> lock(attachment_state_mutex_);
      attachment_state_received_ = false;
    }
    gazebo_detach_pub_->publish(std_msgs::msg::Empty{});
    const bool gazebo_detached = waitForAttachmentState(
      false, attach_confirmation_timeout_);
    if (!gazebo_detached)
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Gazebo did not confirm attachment_state='detached' within %.1f seconds.",
        attach_confirmation_timeout_);
    }

    // Mirror the released state in MoveIt. A missing attached collision
    // object must not prevent the already-requested physical release.
    const bool moveit_detached = move_group_.detachObject(object_id);
    if (!moveit_detached)
    {
      RCLCPP_WARN(
        node_->get_logger(),
        "%s was not attached in MoveIt; Gazebo detach was still requested.",
        object_id.c_str());
    }

    planning_scene_interface_.removeCollisionObjects({object_id});
    rclcpp::sleep_for(std::chrono::milliseconds(100));
    addPickBoxObstacle(
      release_pose.pose.position.x,
      release_pose.pose.position.y,
      0.05);

    RCLCPP_INFO(
      node_->get_logger(),
      "Detached %s and placed its MoveIt scene object on the table.",
      object_id.c_str());
    return gazebo_detached;
  }

  bool UR3MotionInterface::prepareNextTrial()
  {
    const std::string object_id = "pick_box";

    // A failed trial may leave either Gazebo's fixed joint or MoveIt's
    // attached object active. Always release both representations before the
    // test runner teleports the one reusable box to its next random pose.
    gazebo_detach_pub_->publish(std_msgs::msg::Empty{});
    rclcpp::sleep_for(std::chrono::milliseconds(300));

    const bool moveit_detached = move_group_.detachObject(object_id);
    if (!moveit_detached)
    {
      RCLCPP_DEBUG(
        node_->get_logger(),
        "%s was not attached in MoveIt while preparing the next trial.",
        object_id.c_str());
    }

    planning_scene_interface_.removeCollisionObjects({object_id});
    rclcpp::sleep_for(std::chrono::milliseconds(100));

    RCLCPP_INFO(
      node_->get_logger(),
      "Prepared the next trial: Gazebo joint released and stale MoveIt "
      "pick_box object removed.");
    return true;
  }

  bool UR3MotionInterface::moveCartesian(
    double x_offset,
    double y_offset,
    double z_offset,
    bool allow_pose_fallback)
  {
    if (std::abs(x_offset) < 1e-6 &&
        std::abs(y_offset) < 1e-6 &&
        std::abs(z_offset) < 1e-6)
    {
      RCLCPP_ERROR(node_->get_logger(), "At least one Cartesian offset must be non-zero.");
      return false;
    }

    const std::string end_effector_link = "gripper_tcp";
    const auto current_pose = move_group_.getCurrentPose(end_effector_link);
    geometry_msgs::msg::Pose target_pose = current_pose.pose;
    target_pose.position.x += x_offset;
    target_pose.position.y += y_offset;
    target_pose.position.z += z_offset;

    std::vector<geometry_msgs::msg::Pose> waypoints{target_pose};
    moveit_msgs::msg::RobotTrajectory trajectory_msg;

    constexpr double eef_step = 0.005;
    constexpr double required_fraction = 0.99;

    const double fraction = move_group_.computeCartesianPath(
      waypoints,
      eef_step,
      cartesian_jump_threshold_,
      trajectory_msg,
      true);

RCLCPP_INFO(
    node_->get_logger(),
    "Cartesian fraction = %.3f",
    fraction
);

RCLCPP_INFO(
    node_->get_logger(),
    "Generated trajectory points = %zu",
    trajectory_msg.joint_trajectory.points.size()
);

for (size_t i = 0;
     i < trajectory_msg.joint_trajectory.points.size();
     ++i)
{
    const auto& p = trajectory_msg.joint_trajectory.points[i];

    if (p.positions.size() >= 6)
    {
        RCLCPP_INFO(
            node_->get_logger(),
            "Point %zu: q = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
            i,
            p.positions[0],
            p.positions[1],
            p.positions[2],
            p.positions[3],
            p.positions[4],
            p.positions[5]
        );
    }
}
    

    if (fraction < required_fraction)
    {
      if (!allow_pose_fallback)
      {
        RCLCPP_ERROR(
          node_->get_logger(),
          "Strict Cartesian path is only %.1f%% (required %.1f%%); "
          "rejecting the grasp advance without pose fallback.",
          fraction * 100.0, required_fraction * 100.0);
        return false;
      }

      RCLCPP_WARN(
        node_->get_logger(),
        "Straight Cartesian path is only %.1f%% (required %.1f%%); "
        "falling back to pose planning.",
        fraction * 100.0, required_fraction * 100.0);

      // General Cartesian commands may still use a non-straight fallback.
      // The final grasp advance disables this path because a pose planner can
      // rotate the wrist or take a curved route around the object.
      return moveToPoseGoal(target_pose, end_effector_link);
    }

    double observed_maximum_joint_step = 0.0;
    if (hasUnsafeCartesianJointStep(
        trajectory_msg, cartesian_max_joint_step_,
        observed_maximum_joint_step))
    {
      RCLCPP_ERROR(
        node_->get_logger(),
        "Cartesian path contains a %.3f rad joint step (limit %.3f rad); "
        "rejecting a possible IK-branch jump.",
        observed_maximum_joint_step, cartesian_max_joint_step_);
      return false;
    }

    const auto current_state = move_group_.getCurrentState(2.0);
    if (!current_state)
    {
      RCLCPP_ERROR(node_->get_logger(), "Could not read current robot state for trajectory timing.");
      return false;
    }

    robot_trajectory::RobotTrajectory timed_trajectory(
      move_group_.getRobotModel(), move_group_.getName());
    timed_trajectory.setRobotTrajectoryMsg(*current_state, trajectory_msg);

    trajectory_processing::IterativeParabolicTimeParameterization time_parameterization;
    constexpr double lift_velocity_scaling = 0.05;
    constexpr double lift_acceleration_scaling = 0.05;

    if (!time_parameterization.computeTimeStamps(
          timed_trajectory, lift_velocity_scaling, lift_acceleration_scaling))
    {
      RCLCPP_ERROR(node_->get_logger(), "Failed to time-parameterize Cartesian lift.");
      return false;
    }

    timed_trajectory.getRobotTrajectoryMsg(trajectory_msg);

    RCLCPP_INFO(
      node_->get_logger(),
      "Executing Cartesian move: dx=%+.3f, dy=%+.3f, dz=%+.3f m; "
      "maximum sampled joint step %.3f rad.",
      x_offset, y_offset, z_offset, observed_maximum_joint_step);

    const auto execution_result = move_group_.execute(trajectory_msg);
    if (execution_result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(node_->get_logger(), "Cartesian motion execution failed.");
      return false;
    }

    RCLCPP_INFO(node_->get_logger(), "Cartesian motion succeeded.");
    return true;
  }

  bool UR3MotionInterface::moveToXY(double target_x, double target_y)
  {
    const std::string end_effector_link = "gripper_tcp";
    const auto current_pose = move_group_.getCurrentPose(end_effector_link);
    geometry_msgs::msg::Pose target_pose = current_pose.pose;

    // X and Y are absolute coordinates in base_link. Keep the lifted height
    // and tool orientation exactly as reported by the current TCP pose.
    target_pose.position.x = target_x;
    target_pose.position.y = target_y;

    RCLCPP_INFO(
      node_->get_logger(),
      "Planning absolute XY move: (%.3f, %.3f, %.3f) -> "
      "(%.3f, %.3f, %.3f), keeping current Z.",
      current_pose.pose.position.x,
      current_pose.pose.position.y,
      current_pose.pose.position.z,
      target_pose.position.x,
      target_pose.position.y,
      target_pose.position.z);

    // Prefer a straight horizontal transport path. moveCartesian() will fall
    // back to the pose planner only if the straight path is not feasible.
    return moveCartesian(
      target_pose.position.x - current_pose.pose.position.x,
      target_pose.position.y - current_pose.pose.position.y,
      0.0);
  }

  void UR3MotionInterface::stop()
  {
    move_group_.stop();
  }

  void UR3MotionInterface::addTableObstacle()
  {
    moveit_msgs::msg::CollisionObject collision_object;
    // Gazebo places a 1.0 x 0.7 x 0.05 m table at world (0.5, 0, 0.75)
    // while the robot base is at world (0, 0, 0.775). Expressing the same
    // surface in base_link gives centre (0.5, 0, -0.025).
    collision_object.header.frame_id = "base_link";
    collision_object.id = "table";

    const double table_length = 1.0;
    const double table_width = 0.7;
    const double table_thickness = 0.05;
    const double table_center_x = 0.5;
    const double table_center_z = -0.025;

    // Keep the existing fixed wall geometry unchanged. These constants are
    // intentionally independent of the corrected table dimensions.
    const double wall_thickness = 0.02;
    const double wall_height = 0.8;
    const double back_wall_x = -0.2;
    const double side_wall_y = 0.5;
    const double front_wall_x = 0.6;
    const double side_wall_length = front_wall_x - back_wall_x;
    const double side_wall_x = (front_wall_x + back_wall_x) / 2.0;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = table_length;    // dimension along X axis
    primitive.dimensions[1] = table_width;     // dimension along Y axis
    primitive.dimensions[2] = table_thickness; // thickness along Z axis

    geometry_msgs::msg::Pose box_pose;
    box_pose.orientation.w = 1.0;
    box_pose.position.x = table_center_x;
    box_pose.position.y = 0.0;
    box_pose.position.z = table_center_z;

    collision_object.primitives.push_back(primitive);
    collision_object.primitive_poses.push_back(box_pose);
    collision_object.operation = collision_object.ADD;

    shape_msgs::msg::SolidPrimitive back_wall_primitive;
    back_wall_primitive.type = back_wall_primitive.BOX;
    back_wall_primitive.dimensions.resize(3);
    back_wall_primitive.dimensions[0] = wall_thickness;      // thickness along X axis
    back_wall_primitive.dimensions[1] = side_wall_y * 2.0;   // 0.5m offset on both sides
    back_wall_primitive.dimensions[2] = wall_height;         // height along Z axis

    shape_msgs::msg::SolidPrimitive side_wall_primitive;
    side_wall_primitive.type = side_wall_primitive.BOX;
    side_wall_primitive.dimensions.resize(3);
    side_wall_primitive.dimensions[0] = side_wall_length; // from back wall to table's front edge
    side_wall_primitive.dimensions[1] = wall_thickness;   // thickness along Y axis
    side_wall_primitive.dimensions[2] = wall_height;      // height along Z axis

    // Add back wall behind the robot
    moveit_msgs::msg::CollisionObject back_wall;
    back_wall.header.frame_id = move_group_.getPlanningFrame();
    back_wall.id = "back_wall";

    geometry_msgs::msg::Pose back_wall_pose;
    back_wall_pose.orientation.w = 1.0;
    back_wall_pose.position.x = back_wall_x;
    back_wall_pose.position.y = 0.0;
    back_wall_pose.position.z = wall_height / 2.0;

    back_wall.primitives.push_back(back_wall_primitive);
    back_wall.primitive_poses.push_back(back_wall_pose);
    back_wall.operation = back_wall.ADD;

    // Thêm tường bên trái
    moveit_msgs::msg::CollisionObject left_wall;
    left_wall.header.frame_id = move_group_.getPlanningFrame();
    left_wall.id = "left_wall";

    geometry_msgs::msg::Pose left_wall_pose;
    left_wall_pose.orientation.w = 1.0;
    left_wall_pose.position.x = side_wall_x;
    left_wall_pose.position.y = side_wall_y;
    left_wall_pose.position.z = wall_height / 2.0;

    left_wall.primitives.push_back(side_wall_primitive);
    left_wall.primitive_poses.push_back(left_wall_pose);
    left_wall.operation = left_wall.ADD;

    // Thêm tường bên phải
    moveit_msgs::msg::CollisionObject right_wall;
    right_wall.header.frame_id = move_group_.getPlanningFrame();
    right_wall.id = "right_wall";

    geometry_msgs::msg::Pose right_wall_pose;
    right_wall_pose.orientation.w = 1.0;
    right_wall_pose.position.x = side_wall_x;
    right_wall_pose.position.y = -side_wall_y;
    right_wall_pose.position.z = wall_height / 2.0;

    right_wall.primitives.push_back(side_wall_primitive);
    right_wall.primitive_poses.push_back(right_wall_pose);
    right_wall.operation = right_wall.ADD;

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.push_back(collision_object);
    collision_objects.push_back(back_wall);
    collision_objects.push_back(left_wall);
    collision_objects.push_back(right_wall);

    RCLCPP_INFO(node_->get_logger(), "Adding table and walls into the world...");
    planning_scene_interface_.applyCollisionObjects(collision_objects);
  }

  void UR3MotionInterface::addPickBoxObstacle(double x, double y, double z)
  {
    moveit_msgs::msg::CollisionObject pick_box;
    // Coordinates below are relative to the robot mounting frame.
    pick_box.header.frame_id = "base_link";
    pick_box.id = "pick_box";

    const auto add_cylinder = [&pick_box, x, y, z](
      double height, double radius, double z_offset)
    {
      shape_msgs::msg::SolidPrimitive primitive;
      primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
      primitive.dimensions.resize(2);
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_HEIGHT] = height;
      primitive.dimensions[shape_msgs::msg::SolidPrimitive::CYLINDER_RADIUS] = radius;

      geometry_msgs::msg::Pose pose;
      pose.orientation.w = 1.0;
      pose.position.x = x;
      pose.position.y = y;
      pose.position.z = z + z_offset;
      pick_box.primitives.push_back(primitive);
      pick_box.primitive_poses.push_back(pose);
    };

    // The object reference origin remains 0.05 m above the bottle bottom,
    // matching the old box coordinates used by the clients and test runner.
    add_cylinder(0.160, 0.050, 0.030);  // 0.10 m main body

    shape_msgs::msg::SolidPrimitive shoulder;
    shoulder.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    shoulder.dimensions = {0.050};
    geometry_msgs::msg::Pose shoulder_pose;
    shoulder_pose.orientation.w = 1.0;
    shoulder_pose.position.x = x;
    shoulder_pose.position.y = y;
    shoulder_pose.position.z = z + 0.110;
    pick_box.primitives.push_back(shoulder);
    pick_box.primitive_poses.push_back(shoulder_pose);

    add_cylinder(0.040, 0.022, 0.1750);  // neck
    add_cylinder(0.015, 0.030, 0.2025);  // metal lip
    pick_box.operation = moveit_msgs::msg::CollisionObject::ADD;

    RCLCPP_INFO(
      node_->get_logger(),
      "Adding bottle collision object pick_box at (%.3f, %.3f, %.3f)...",
      x, y, z);
    planning_scene_interface_.applyCollisionObject(pick_box);
  }

  void UR3MotionInterface::calibObjectHeightEyeInHand(double depth_m, double camera_z_mount_offset)
  {
    geometry_msgs::msg::PoseStamped current_pose = move_group_.getCurrentPose();
    double ee_z = current_pose.pose.position.z;
    double object_height = ee_z - camera_z_mount_offset - depth_m;
    RCLCPP_INFO_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000, "Current EE Z: %.3f, Depth: %.3f, Object Height: %.3f", ee_z, depth_m, object_height);
  }

} // namespace ur3_moveit_control
