from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    common_arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("llm_vision_planner"), "config", "llm_vision_planner.yaml"]
            ),
        ),
        DeclareLaunchArgument("trial_id", default_value="unset"),
        DeclareLaunchArgument(
            "output_csv",
            default_value="fine_tuning/datasets/calibration_vision_error.csv",
        ),
        DeclareLaunchArgument("object_id", default_value="obj-1"),
        DeclareLaunchArgument("object_label", default_value="chair"),
        DeclareLaunchArgument("object_vicon_topic", default_value="/vicon/chair1/chair1"),
        DeclareLaunchArgument("object_width_m", default_value="0.0"),
        DeclareLaunchArgument("object_depth_m", default_value="0.0"),
        DeclareLaunchArgument("marker_to_object_json", default_value="{}"),
        DeclareLaunchArgument("vicon_vehicle_topic", default_value="/vicon/Starling2/Starling2"),
        DeclareLaunchArgument("vicon_vehicle_frame_convention", default_value="flu"),
        DeclareLaunchArgument("pose_topic", default_value="/fmu/out/vehicle_odometry"),
        DeclareLaunchArgument("auto_vicon_world_to_ned", default_value="true"),
        DeclareLaunchArgument("vicon_world_to_ned_json", default_value="{}"),
        DeclareLaunchArgument("frame_calibration_samples", default_value="20"),
        DeclareLaunchArgument("calibration_capture_delay_s", default_value="1.0"),
        DeclareLaunchArgument("openai_intent_model", default_value="gpt-5.4-nano"),
    ]

    recorder = Node(
        package="llm_vision_planner",
        executable="vision_error_dataset_generattor.py",
        name="vision_error_dataset_generator",
        output="screen",
        parameters=[
            params_file,
            {
                "trial_id": LaunchConfiguration("trial_id"),
                "output_csv": LaunchConfiguration("output_csv"),
                "object_id": LaunchConfiguration("object_id"),
                "object_label": LaunchConfiguration("object_label"),
                "object_vicon_topic": LaunchConfiguration("object_vicon_topic"),
                "object_width_m": ParameterValue(
                    LaunchConfiguration("object_width_m"), value_type=float
                ),
                "object_depth_m": ParameterValue(
                    LaunchConfiguration("object_depth_m"), value_type=float
                ),
                "marker_to_object_json": LaunchConfiguration("marker_to_object_json"),
                "vicon_vehicle_topic": LaunchConfiguration("vicon_vehicle_topic"),
                "vicon_vehicle_frame_convention": LaunchConfiguration(
                    "vicon_vehicle_frame_convention"
                ),
                "pose_topic": LaunchConfiguration("pose_topic"),
                "auto_vicon_world_to_ned": ParameterValue(
                    LaunchConfiguration("auto_vicon_world_to_ned"), value_type=bool
                ),
                "vicon_world_to_ned_json": LaunchConfiguration("vicon_world_to_ned_json"),
                "frame_calibration_samples": ParameterValue(
                    LaunchConfiguration("frame_calibration_samples"), value_type=int
                ),
            },
        ],
    )

    perception = Node(
        package="llm_vision_planner",
        executable="perception_detection.py",
        name="semantic_obstacle_perception",
        output="screen",
        parameters=[params_file],
    )

    nominal_environment = Node(
        package="llm_vision_planner",
        executable="interactive_mission_gateway.py",
        name="vision_calibration_environment",
        output="screen",
        parameters=[
            params_file,
            {
                "environment": "real",
                "intent_provider": "openai",
                "openai_intent_model": LaunchConfiguration("openai_intent_model"),
                "visualizer": "standard",
                "obs_safety_bracket": "hardcoded",
                "require_mission_state": False,
                "calibration_only": True,
                "auto_calibration_capture": True,
                "calibration_capture_delay_s": ParameterValue(
                    LaunchConfiguration("calibration_capture_delay_s"), value_type=float
                ),
                "pose_topic": LaunchConfiguration("pose_topic"),
            },
        ],
    )

    return LaunchDescription(
        [
            *common_arguments,
            recorder,
            perception,
            TimerAction(period=1.0, actions=[nominal_environment]),
        ]
    )
