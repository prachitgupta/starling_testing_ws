from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import PathJoinSubstitution
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    environment_arg = DeclareLaunchArgument(
        "environment",
        default_value="real",
        description="Mission environment: real uses TFLite/ToF perception; sim uses /llm_vision/sim_obstacles.",
    )
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("llm_vision_planner"), "config", "llm_vision_planner.yaml"]
        ),
        description="YAML file containing llm_vision_planner node parameters",
    )
    goal_x_arg = DeclareLaunchArgument(
        "goal_x",
        default_value="0.0",
        description="Fixed mission goal x coordinate in local NED metres.",
    )
    goal_y_arg = DeclareLaunchArgument(
        "goal_y",
        default_value="1.5",
        description="Fixed mission goal y coordinate in local NED metres.",
    )
    llm_provider_arg = DeclareLaunchArgument(
        "llm_provider",
        default_value="llama",
        description="Planner LLM provider: chatgpt or llama",
    )
    show_rrt_arg = DeclareLaunchArgument(
        "show_rrt",
        default_value="true",
        description="Overlay an RRT expert trajectory on the final verified plot",
    )
    visualizer_arg = DeclareLaunchArgument(
        "visualizer",
        default_value="contraction",
        description="Live plot: standard or contraction",
    )
    land_after_complete_arg = DeclareLaunchArgument(
        "land_after_complete",
        default_value="true",
        description="Land after a successful mission; false holds the final goal in Offboard.",
    )
    interaction_mode_arg = DeclareLaunchArgument(
        "interaction_mode",
        default_value="fixed",
        description="Mission prompt source: fixed preserves the existing pipeline; interactive requires human approval.",
    )
    intent_provider_arg = DeclareLaunchArgument(
        "intent_provider",
        default_value="openai",
        description="Interactive intent parser: openai or mock (mock is for offline simulation only).",
    )
    web_ui_arg = DeclareLaunchArgument(
        "web_ui",
        default_value="true",
        description="Launch the local operator web UI in interactive mode.",
    )
    web_ui_port_arg = DeclareLaunchArgument(
        "web_ui_port",
        default_value="8080",
        description="Local HTTP port for the interactive operator UI.",
    )
    web_ui_host_arg = DeclareLaunchArgument(
        "web_ui_host",
        default_value="127.0.0.1",
        description="Bind address for the interactive operator UI. Set to 0.0.0.0 "
        "to reach it from another device (e.g. a phone) on the same network.",
    )
    use_dataset_scene_arg = DeclareLaunchArgument(
        "use_dataset_scene",
        default_value="false",
        description="Publish a recorded env_ros_commands.csv scene when environment=sim.",
    )
    sim_sample_id_arg = DeclareLaunchArgument(
        "sim_sample_id",
        default_value="4",
        description="env_ros_commands.csv sample used by the optional simulation scene publisher.",
    )
    params_file = LaunchConfiguration("params_file")
    fixed_goal = {
        "goal_x": ParameterValue(LaunchConfiguration("goal_x"), value_type=float),
        "goal_y": ParameterValue(LaunchConfiguration("goal_y"), value_type=float),
    }

    use_real_perception = IfCondition(PythonExpression(["'", LaunchConfiguration("environment"), "' == 'real'"]))
    use_fixed_prompt = IfCondition(PythonExpression(["'", LaunchConfiguration("interaction_mode"), "' == 'fixed'"]))
    use_interactive_prompt = IfCondition(PythonExpression(["'", LaunchConfiguration("interaction_mode"), "' == 'interactive'"]))
    use_interactive_web_ui = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("interaction_mode"),
                "' == 'interactive' and '",
                LaunchConfiguration("web_ui"),
                "'.lower() == 'true'",
            ]
        )
    )
    use_dataset_scene = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("environment"),
                "' == 'sim' and '",
                LaunchConfiguration("use_dataset_scene"),
                "'.lower() == 'true'",
            ]
        )
    )

    semantic_perception = Node(
        package="llm_vision_planner",
        executable="perception_detection.py",
        name="semantic_obstacle_perception",
        output="screen",
        condition=use_real_perception,
        parameters=[params_file, fixed_goal],
    )

    prompt_generator = Node(
        package="llm_vision_planner",
        executable="prompt_generator.py",
        name="prompt_generator",
        output="screen",
        parameters=[
            params_file,
            {
                "environment": LaunchConfiguration("environment"),
                "llm_provider": LaunchConfiguration("llm_provider"),
                **fixed_goal,
            },
        ],
        condition=use_fixed_prompt,
    )

    interactive_gateway = Node(
        package="llm_vision_planner",
        executable="interactive_mission_gateway.py",
        name="interactive_mission_gateway",
        output="screen",
        parameters=[
            params_file,
            {
                "environment": LaunchConfiguration("environment"),
                "intent_provider": LaunchConfiguration("intent_provider"),
                "planner_llm_provider": LaunchConfiguration("llm_provider"),
                "visualizer": LaunchConfiguration("visualizer"),
            },
        ],
        condition=use_interactive_prompt,
    )

    interactive_web_ui = Node(
        package="llm_vision_planner",
        executable="interactive_web_ui.py",
        name="interactive_web_ui",
        output="screen",
        parameters=[
            params_file,
            {
                "port": ParameterValue(LaunchConfiguration("web_ui_port"), value_type=int),
                "host": LaunchConfiguration("web_ui_host"),
                "visualizer": LaunchConfiguration("visualizer"),
            },
        ],
        condition=use_interactive_web_ui,
    )

    dataset_scene = Node(
        package="llm_vision_planner",
        executable="dataset_scene_publisher.py",
        name="dataset_scene_publisher",
        output="screen",
        parameters=[params_file, {"sample_id": ParameterValue(LaunchConfiguration("sim_sample_id"), value_type=int)}],
        condition=use_dataset_scene,
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

    fixed_verifier = Node(
        package="llm_vision_planner",
        executable="verifier.py",
        name="path_verifier",
        output="screen",
        parameters=[params_file],
        condition=use_fixed_prompt,
    )

    interactive_verifier = Node(
        package="llm_vision_planner",
        executable="verifier.py",
        name="path_verifier",
        output="screen",
        parameters=[params_file, {"verified_plan_topic": "/llm_vision/plan_candidate_verified"}],
        condition=use_interactive_prompt,
    )

    control_executor = Node(
        package="llm_vision_planner",
        executable="control_law_executer.py",
        name="control_law_executer",
        output="screen",
        parameters=[
            params_file,
            {
                "land_after_complete": ParameterValue(
                    LaunchConfiguration("land_after_complete"), value_type=bool
                )
            },
        ],
    )

    visualizer = Node(
        package="llm_vision_planner",
        executable="visualize.py",
        name="planner_visualizer",
        output="screen",
        parameters=[params_file, {"show_rrt": ParameterValue(LaunchConfiguration("show_rrt"), value_type=bool)}],
        condition=UnlessCondition(PythonExpression(["'", LaunchConfiguration("visualizer"), "' == 'contraction'"])),
    )

    contraction_visualizer = Node(
        package="llm_vision_planner",
        executable="verify_contraction.py",
        name="verify_contraction",
        output="screen",
        parameters=[params_file, {"environment": LaunchConfiguration("environment")}],
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("visualizer"), "' == 'contraction'"])),
    )

    return LaunchDescription(
        [
            environment_arg,
            params_file_arg,
            goal_x_arg,
            goal_y_arg,
            llm_provider_arg,
            show_rrt_arg,
            visualizer_arg,
            land_after_complete_arg,
            interaction_mode_arg,
            intent_provider_arg,
            web_ui_arg,
            web_ui_port_arg,
            web_ui_host_arg,
            use_dataset_scene_arg,
            sim_sample_id_arg,
            semantic_perception,
            dataset_scene,
            llm_planner,
            TimerAction(period=2.0, actions=[prompt_generator]),
            interactive_gateway,
            interactive_web_ui,
            refinement,
            fixed_verifier,
            interactive_verifier,
            control_executor,
            visualizer,
            contraction_visualizer,
        ]
    )
