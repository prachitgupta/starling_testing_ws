from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    mode_arg = DeclareLaunchArgument(
        "mode",
        default_value="semantic",
        description="Obstacle source mode for prompt generator: semantic or normal",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("llm_vision_planner"), "config", "llm_vision_planner.yaml"]
        ),
        description="YAML file containing llm_vision_planner node parameters",
    )
    llm_provider_arg = DeclareLaunchArgument(
        "llm_provider",
        default_value="chatgpt",
        description="Planner LLM provider: chatgpt or llama",
    )
    show_rrt_arg = DeclareLaunchArgument(
        "show_rrt",
        default_value="false",
        description="Overlay an RRT expert trajectory on the final verified plot",
    )
    params_file = LaunchConfiguration("params_file")

    use_semantic = IfCondition(PythonExpression(["'", LaunchConfiguration("mode"), "' == 'semantic'"]))
    use_normal = UnlessCondition(PythonExpression(["'", LaunchConfiguration("mode"), "' == 'semantic'"]))

    semantic_perception = Node(
        package="llm_vision_planner",
        executable="perception_detection.py",
        name="semantic_obstacle_perception",
        output="screen",
        condition=use_semantic,
        parameters=[params_file],
    )

    normal_perception = Node(
        package="llm_vision_planner",
        executable="perception.py",
        name="obstacle_perception",
        output="screen",
        condition=use_normal,
        parameters=[params_file],
    )

    prompt_generator = Node(
        package="llm_vision_planner",
        executable="prompt_generator.py",
        name="prompt_generator",
        output="screen",
        parameters=[params_file, {"mode": LaunchConfiguration("mode"), "llm_provider": LaunchConfiguration("llm_provider")}],
    )

    llm_planner = Node(
        package="llm_vision_planner",
        executable="llm_planner.py",
        name="llm_planner",
        output="screen",
        parameters=[params_file, {"llm_provider": LaunchConfiguration("llm_provider")}],
    )

    refinement = Node(
        package="llm_vision_planner",
        executable="refinment.py",
        name="path_refinement",
        output="screen",
        parameters=[params_file],
    )

    verifier = Node(
        package="llm_vision_planner",
        executable="verifier.py",
        name="path_verifier",
        output="screen",
        parameters=[params_file],
    )

    visualizer = Node(
        package="llm_vision_planner",
        executable="visualize.py",
        name="planner_visualizer",
        output="screen",
        parameters=[params_file, {"show_rrt": ParameterValue(LaunchConfiguration("show_rrt"), value_type=bool)}],
    )

    return LaunchDescription(
        [
            mode_arg,
            params_file_arg,
            llm_provider_arg,
            show_rrt_arg,
            semantic_perception,
            normal_perception,
            llm_planner,
            TimerAction(period=2.0, actions=[prompt_generator]),
            refinement,
            verifier,
            visualizer,
        ]
    )
