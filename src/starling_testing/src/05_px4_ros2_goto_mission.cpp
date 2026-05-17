#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>

#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/mode_executor.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <px4_ros2/odometry/local_position.hpp>
#include <rclcpp/rclcpp.hpp>

class VoxlGotoMissionMode : public px4_ros2::ModeBase
{
public:
  explicit VoxlGotoMissionMode(rclcpp::Node & node)
  : ModeBase(node, Settings{"VOXL Goto Mission"}.preventArming(false)),
    node_(node),
    goto_(std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this)),
    local_position_(std::make_shared<px4_ros2::OdometryLocalPosition>(*this))
  {
    target_.x() = static_cast<float>(node_.declare_parameter<double>("x", 0.0));
    target_.y() = static_cast<float>(node_.declare_parameter<double>("y", 2.0));
    target_.z() = static_cast<float>(node_.declare_parameter<double>("z", -1.5));
    speed_ = static_cast<float>(node_.declare_parameter<double>("speed", 1.0));
    vertical_speed_ = static_cast<float>(node_.declare_parameter<double>("vertical_speed", 0.8));
    accept_m_ = static_cast<float>(node_.declare_parameter<double>("accept_m", 0.4));
    const double heading = node_.declare_parameter<double>("heading", std::numeric_limits<double>::quiet_NaN());
    if (std::isfinite(heading)) {
      heading_ = static_cast<float>(heading);
    }
  }

  void onActivate() override { reached_ = false; }

  void updateSetpoint(float dt_s) override
  {
    (void)dt_s;
    goto_->update(target_, heading_, speed_, vertical_speed_);

    if (reached_ || !local_position_->positionXYValid() || !local_position_->positionZValid()) {
      return;
    }

    const float error_m = (local_position_->positionNed() - target_).norm();
    RCLCPP_INFO_THROTTLE(
      node_.get_logger(), *node_.get_clock(), 1000,
      "goto target=(%.2f %.2f %.2f), error=%.2f m",
      target_.x(), target_.y(), target_.z(), error_m);

    if (error_m <= accept_m_) {
      reached_ = true;
      RCLCPP_INFO(node_.get_logger(), "Waypoint reached. Completing mode so executor can land.");
      completed(px4_ros2::Result::Success);
    }
  }

private:
  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_;
  std::shared_ptr<px4_ros2::OdometryLocalPosition> local_position_;
  Eigen::Vector3f target_{0.0f, 2.0f, -1.5f};
  std::optional<float> heading_{};
  float speed_{1.0f};
  float vertical_speed_{0.8f};
  float accept_m_{0.4f};
  bool reached_{false};
};

class VoxlGotoMissionExecutor : public px4_ros2::ModeExecutorBase
{
public:
  explicit VoxlGotoMissionExecutor(px4_ros2::ModeBase & mode)
  : ModeExecutorBase(
      px4_ros2::ModeExecutorBase::Settings{}.activate(
        px4_ros2::ModeExecutorBase::Settings::Activation::ActivateImmediately),
      mode),
    node_(mode.node())
  {
    takeoff_altitude_amsl_ = node_.declare_parameter<double>(
      "takeoff_altitude_amsl", std::numeric_limits<double>::quiet_NaN());
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
    RCLCPP_INFO(node_.get_logger(), "Mission executor deactivated (%d)", static_cast<int>(reason));
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
    RCLCPP_INFO(node_.get_logger(), "Takeoff complete. Switching to custom goto mode.");
    scheduleMode(ownedMode().id(), [this](px4_ros2::Result mode_r) {
      RCLCPP_INFO(node_.get_logger(), "Goto mode finished: %s. Landing.", px4_ros2::resultToString(mode_r));
      land([this](px4_ros2::Result land_r) {
        RCLCPP_INFO(node_.get_logger(), "Land command finished: %s", px4_ros2::resultToString(land_r));
        waitUntilDisarmed([](px4_ros2::Result) { rclcpp::shutdown(); });
      });
    });
  }

  rclcpp::Node & node_;
  double takeoff_altitude_amsl_{std::numeric_limits<double>::quiet_NaN()};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  using Node = px4_ros2::NodeWithModeExecutor<VoxlGotoMissionExecutor, VoxlGotoMissionMode>;
  rclcpp::spin(std::make_shared<Node>("voxl_px4_ros2_goto_mission_cpp"));
  rclcpp::shutdown();
  return 0;
}

