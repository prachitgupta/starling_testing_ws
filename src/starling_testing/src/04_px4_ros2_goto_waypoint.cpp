#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <memory>
#include <string>

#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_ros2/components/mode.hpp>
#include <px4_ros2/components/node_with_mode.hpp>
#include <px4_ros2/control/setpoint_types/multicopter/goto.hpp>
#include <rclcpp/rclcpp.hpp>

class VoxlGotoMode : public px4_ros2::ModeBase
{
public:
  explicit VoxlGotoMode(rclcpp::Node & node)
  : ModeBase(node, Settings{"VOXL C++ Goto Smoke Test"}.preventArming(false)),
    node_(node),
    goto_(std::make_shared<px4_ros2::MulticopterGotoSetpointType>(*this))
  {
    x_ = node_.declare_parameter<double>("x", 0.0);
    y_ = node_.declare_parameter<double>("y", 2.0);
    z_ = node_.declare_parameter<double>("z", -1.5);
    speed_ = node_.declare_parameter<double>("speed", 1.0);
    heading_ = node_.declare_parameter<double>("heading", std::numeric_limits<double>::quiet_NaN());
    pose_topic_ = node_.declare_parameter<std::string>("pose_topic", "/fmu/out/vehicle_odometry");
    pose_timeout_s_ = node_.declare_parameter<double>("pose_timeout_s", 0.5);
    odometry_sub_ = node_.create_subscription<px4_msgs::msg::VehicleOdometry>(
      pose_topic_, rclcpp::SensorDataQoS(),
      [this](const px4_msgs::msg::VehicleOdometry::SharedPtr msg) {
        if (msg->pose_frame != px4_msgs::msg::VehicleOdometry::POSE_FRAME_NED) {
          RCLCPP_WARN_THROTTLE(
            node_.get_logger(), *node_.get_clock(), 2000,
            "ignoring %s: pose_frame=%u, expected NED", pose_topic_.c_str(), msg->pose_frame);
          return;
        }
        last_odometry_time_ = node_.now();
        have_odometry_ = true;
      });
  }

  void updateSetpoint(float dt_s) override
  {
    (void)dt_s;
    if (!odometryFresh()) {
      RCLCPP_WARN_THROTTLE(
        node_.get_logger(), *node_.get_clock(), 2000,
        "waiting for fresh PX4 odometry on %s", pose_topic_.c_str());
      return;
    }
    Eigen::Vector3f target{static_cast<float>(x_), static_cast<float>(y_), static_cast<float>(z_)};
    float heading = std::isfinite(heading_) ? static_cast<float>(heading_) : NAN;
    goto_->update(target, heading, static_cast<float>(speed_));
  }

private:
  bool odometryFresh() const
  {
    return have_odometry_ && (node_.now() - last_odometry_time_).seconds() <= pose_timeout_s_;
  }

  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_;
  rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
  rclcpp::Time last_odometry_time_{0, 0, RCL_ROS_TIME};
  std::string pose_topic_{"/fmu/out/vehicle_odometry"};
  double x_{}, y_{}, z_{}, speed_{}, heading_{};
  double pose_timeout_s_{0.5};
  bool have_odometry_{false};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  using Node = px4_ros2::NodeWithMode<VoxlGotoMode>;
  rclcpp::spin(std::make_shared<Node>("voxl_px4_ros2_goto_waypoint_cpp"));
  rclcpp::shutdown();
  return 0;
}
