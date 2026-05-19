#include <Eigen/Core>
#include <json/json.h>

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/mode_executor.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class TrajectoryFollowerMode : public px4_ros2::ModeBase
{
public:
  explicit TrajectoryFollowerMode(rclcpp::Node & node)
  : ModeBase(node, Settings{"LLM Vision Trajectory Follower"}.preventArming(false)),
    node_(node),
    goto_(std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this)),
    local_position_(std::make_shared<px4_ros2::OdometryLocalPosition>(*this))
  {
    plan_topic_ = node_.declare_parameter<std::string>("plan_topic", "/llm_vision/plan_verified");
    speed_ = static_cast<float>(node_.declare_parameter<double>("speed", 1.0));
    vertical_speed_ = static_cast<float>(node_.declare_parameter<double>("vertical_speed", 0.8));
    accept_m_ = static_cast<float>(node_.declare_parameter<double>("accept_m", 0.35));
    start_accept_m_ = static_cast<float>(node_.declare_parameter<double>("start_accept_m", 0.75));
    hold_after_mission_ = node_.declare_parameter<bool>("hold_after_mission", false);
    heading_ = node_.declare_parameter<double>("heading", std::numeric_limits<double>::quiet_NaN());

    verified_plan_sub_ = node_.create_subscription<std_msgs::msg::String>(
      plan_topic_, 10,
      std::bind(&TrajectoryFollowerMode::verifiedPlanCallback, this, std::placeholders::_1));
  }

  void onActivate() override
  {
    accepting_plans_ = true;
    plan_latched_ = false;
    mission_started_ = false;
    mission_complete_ = false;
    pending_plan_available_ = false;
    hold_position_.reset();
    waypoints_.clear();
    waypoint_index_ = 0U;
    RCLCPP_INFO(
      node_.get_logger(),
      "Trajectory follower active. Holding current PX4 local position until a verified plan is latched.");
  }

  void updateSetpoint(float dt_s) override
  {
    (void)dt_s;
    if (!local_position_->positionXYValid() || !local_position_->positionZValid()) {
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 3000,
        "Waiting for valid PX4 local position before holding or tracking.");
      return;
    }

    if (!hold_position_.has_value()) {
      hold_position_ = local_position_->positionNed();
      RCLCPP_INFO(
        node_.get_logger(), "Captured post-takeoff hold position: (%.2f %.2f %.2f)",
        hold_position_->x(), hold_position_->y(), hold_position_->z());
    }

    if (!plan_latched_) {
      if (pending_plan_available_) {
        latchPendingPlanIfSafe();
      }

      if (!plan_latched_) {
        const float heading = std::isfinite(heading_) ? static_cast<float>(heading_) : NAN;
        goto_->update(*hold_position_, heading, speed_, vertical_speed_);
        RCLCPP_WARN_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 3000,
          "Holding at post-takeoff position while waiting for passed=true verified trajectory on %s.",
          plan_topic_.c_str());
        return;
      }
    }

    if (mission_complete_) {
      if (hold_after_mission_ && !waypoints_.empty()) {
        const float heading = std::isfinite(heading_) ? static_cast<float>(heading_) : NAN;
        goto_->update(waypoints_.back(), heading, speed_, vertical_speed_);
        RCLCPP_INFO_THROTTLE(
          node_.get_logger(), *node_.get_clock(), 3000,
          "Holding final verified waypoint because hold_after_mission=true.");
      }
      return;
    }

    const Eigen::Vector3f & target = waypoints_[waypoint_index_];
    const float heading = std::isfinite(heading_) ? static_cast<float>(heading_) : NAN;
    goto_->update(target, heading, speed_, vertical_speed_);

    const float error_m = (local_position_->positionNed() - target).norm();
    RCLCPP_INFO_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "Tracking waypoint %zu/%zu target=(%.2f %.2f %.2f), error=%.2f m",
      waypoint_index_ + 1, waypoints_.size(), target.x(), target.y(), target.z(), error_m);

    if (error_m > accept_m_) {
      return;
    }

    if (waypoint_index_ + 1U < waypoints_.size()) {
      ++waypoint_index_;
      RCLCPP_INFO(
        node_.get_logger(), "Advancing to waypoint %zu/%zu.", waypoint_index_ + 1, waypoints_.size());
      return;
    }

    mission_complete_ = true;
    accepting_plans_ = false;
    if (hold_after_mission_) {
      RCLCPP_INFO(node_.get_logger(), "Final waypoint reached. Holding final setpoint.");
      return;
    }

    RCLCPP_INFO(node_.get_logger(), "Final waypoint reached. Completing trajectory follower mode.");
    completed(px4_ros2::Result::Success);
  }

private:
  void verifiedPlanCallback(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!accepting_plans_) {
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 3000,
        "Ignoring verified-plan message because trajectory follower mode is not accepting plans.");
      return;
    }

    if (plan_latched_ || mission_started_) {
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 3000,
        "Ignoring verified-plan update because a trajectory is already latched/executing.");
      return;
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;
    std::istringstream stream(msg->data);
    if (!Json::parseFromStream(builder, stream, &root, &errors)) {
      RCLCPP_ERROR(node_.get_logger(), "Failed to parse verified plan JSON: %s", errors.c_str());
      return;
    }

    if (!root.get("passed", false).asBool()) {
      RCLCPP_WARN(node_.get_logger(), "Ignoring verified-plan message because passed=false.");
      return;
    }

    const Json::Value waypoints = root["waypoints"];
    if (!waypoints.isArray() || waypoints.empty()) {
      RCLCPP_WARN(node_.get_logger(), "Ignoring verified-plan message because it has no waypoints.");
      return;
    }

    std::vector<Eigen::Vector3f> parsed_waypoints;
    parsed_waypoints.reserve(waypoints.size());
    for (const auto & waypoint : waypoints) {
      if (!waypoint.isObject() || !waypoint.isMember("x") || !waypoint.isMember("y") || !waypoint.isMember("z")) {
        RCLCPP_WARN(node_.get_logger(), "Skipping malformed waypoint entry in verified plan.");
        continue;
      }

      parsed_waypoints.emplace_back(
        waypoint["x"].asFloat(), waypoint["y"].asFloat(), waypoint["z"].asFloat());
    }

    if (parsed_waypoints.empty()) {
      RCLCPP_WARN(node_.get_logger(), "Ignoring verified-plan message because no waypoint entries were usable.");
      return;
    }

    pending_waypoints_ = std::move(parsed_waypoints);
    pending_plan_available_ = true;
    RCLCPP_INFO(
      node_.get_logger(),
      "Received passed=true verified plan with %zu waypoints; waiting to latch against current hold pose.",
      pending_waypoints_.size());
  }

  void latchPendingPlanIfSafe()
  {
    if (!hold_position_.has_value() || pending_waypoints_.empty()) {
      return;
    }

    const float start_error_m = (pending_waypoints_.front() - *hold_position_).norm();
    if (start_error_m > start_accept_m_) {
      RCLCPP_ERROR(
        node_.get_logger(),
        "Rejecting verified plan: first waypoint is %.2f m from post-takeoff hold position; limit is %.2f m.",
        start_error_m, start_accept_m_);
      pending_waypoints_.clear();
      pending_plan_available_ = false;
      return;
    }

    waypoints_ = std::move(pending_waypoints_);
    pending_plan_available_ = false;
    plan_latched_ = true;
    mission_started_ = true;
    waypoint_index_ = 0U;
    mission_complete_ = false;
    accepting_plans_ = false;
    RCLCPP_INFO(
      node_.get_logger(),
      "Latched verified trajectory with %zu waypoints. Future plan updates will be ignored.",
      waypoints_.size());
  }

  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> local_position_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr verified_plan_sub_;
  std::optional<Eigen::Vector3f> hold_position_{};
  std::vector<Eigen::Vector3f> pending_waypoints_{};
  std::vector<Eigen::Vector3f> waypoints_{};
  std::size_t waypoint_index_{0U};
  std::string plan_topic_{"/llm_vision/plan_verified"};
  float speed_{1.0f};
  float vertical_speed_{0.8f};
  float accept_m_{0.35f};
  float start_accept_m_{0.75f};
  double heading_{std::numeric_limits<double>::quiet_NaN()};
  bool accepting_plans_{false};
  bool pending_plan_available_{false};
  bool plan_latched_{false};
  bool mission_started_{false};
  bool mission_complete_{false};
  bool hold_after_mission_{false};
};

class TrajectoryFollowerExecutor : public px4_ros2::ModeExecutorBase
{
public:
  explicit TrajectoryFollowerExecutor(px4_ros2::ModeBase & mode)
  : ModeExecutorBase(
      px4_ros2::ModeExecutorBase::Settings{}.activate(
        px4_ros2::ModeExecutorBase::Settings::Activation::ActivateImmediately),
      mode),
    node_(mode.node())
  {
    takeoff_altitude_amsl_ = node_.declare_parameter<double>(
      "takeoff_altitude_amsl", std::numeric_limits<double>::quiet_NaN());
    land_after_mission_ = node_.declare_parameter<bool>("land_after_mission", true);
  }

  void onActivate() override
  {
    RCLCPP_INFO(node_.get_logger(), "Waiting for PX4 pre-arm checks.");
    waitReadyToArm([this](px4_ros2::Result r) {
      if (r != px4_ros2::Result::Success) {
        RCLCPP_ERROR(node_.get_logger(), "Pre-arm checks failed: %s", px4_ros2::resultToString(r));
        return;
      }
      RCLCPP_INFO(node_.get_logger(), "Pre-arm checks passed. Arming.");
      arm([this](px4_ros2::Result arm_r) { onArmed(arm_r); });
    });
  }

  void onDeactivate(DeactivateReason reason) override
  {
    RCLCPP_INFO(node_.get_logger(), "Trajectory follower executor deactivated (%d)", static_cast<int>(reason));
  }

private:
  void onArmed(px4_ros2::Result r)
  {
    if (r != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Arm failed: %s", px4_ros2::resultToString(r));
      return;
    }

    RCLCPP_INFO(node_.get_logger(), "Armed. Starting PX4 takeoff.");
    takeoff([this](px4_ros2::Result takeoff_r) { onTakeoffDone(takeoff_r); },
      static_cast<float>(takeoff_altitude_amsl_));
  }

  void onTakeoffDone(px4_ros2::Result r)
  {
    if (r != px4_ros2::Result::Success) {
      RCLCPP_ERROR(node_.get_logger(), "Takeoff failed: %s", px4_ros2::resultToString(r));
      return;
    }

    RCLCPP_INFO(node_.get_logger(), "Takeoff complete. Switching to verified trajectory follower mode.");
    scheduleMode(ownedMode().id(), [this](px4_ros2::Result mode_r) { onModeDone(mode_r); });
  }

  void onModeDone(px4_ros2::Result r)
  {
    RCLCPP_INFO(node_.get_logger(), "Verified trajectory follower finished: %s", px4_ros2::resultToString(r));
    if (!land_after_mission_) {
      rclcpp::shutdown();
      return;
    }

    RCLCPP_INFO(node_.get_logger(), "Landing after verified trajectory.");
    land([this](px4_ros2::Result land_r) {
      RCLCPP_INFO(node_.get_logger(), "Land command finished: %s", px4_ros2::resultToString(land_r));
      waitUntilDisarmed([](px4_ros2::Result) { rclcpp::shutdown(); });
    });
  }

  rclcpp::Node & node_;
  double takeoff_altitude_amsl_{std::numeric_limits<double>::quiet_NaN()};
  bool land_after_mission_{true};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  using Node = px4_ros2::NodeWithModeExecutor<TrajectoryFollowerExecutor, TrajectoryFollowerMode>;
  rclcpp::spin(std::make_shared<Node>("llm_vision_trajectory_follower"));
  rclcpp::shutdown();
  return 0;
}
