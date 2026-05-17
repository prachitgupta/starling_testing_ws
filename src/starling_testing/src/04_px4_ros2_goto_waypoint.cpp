#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <memory>

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
  }

  void updateSetpoint(float dt_s) override
  {
    (void)dt_s;
    Eigen::Vector3f target{static_cast<float>(x_), static_cast<float>(y_), static_cast<float>(z_)};
    float heading = std::isfinite(heading_) ? static_cast<float>(heading_) : NAN;
    goto_->update(target, heading, static_cast<float>(speed_));
  }

private:
  rclcpp::Node & node_;
  std::shared_ptr<px4_ros2::MulticopterGotoSetpointType> goto_;
  double x_{}, y_{}, z_{}, speed_{}, heading_{};
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  using Node = px4_ros2::NodeWithMode<VoxlGotoMode>;
  rclcpp::spin(std::make_shared<Node>("voxl_px4_ros2_goto_waypoint_cpp"));
  rclcpp::shutdown();
  return 0;
}

