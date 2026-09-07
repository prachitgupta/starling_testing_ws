# Finding an expert: Semantic Theta* and RRT

Branch: `finding_expert`. This workspace isolates the existing Semantic Theta*
expert, with RRT as the comparison baseline, instruction-data generation, LoRA
fine-tuning, and the shared QP/calibration/plot pipeline. The Python algorithms
and CLI entry points are reused unchanged. Hardware launch/control/perception,
flight logs, papers, and unrelated historical datasets are removed from this
branch. The ROS package and required message dependency remain buildable because
the RRT generator imports the existing ROS prompt/refinement/verifier modules.

Read these three guides in order:

1. **This README:** clone/build, generate either expert dataset, and serve the
   corresponding trained adapter.
2. **[NCSA_COMMANDS.md](NCSA_COMMANDS.md):** upload, train with PEFT or Unsloth,
   monitor, package, and download either adapter on DeltaAI.
3. **[Calibration README](src/llm_vision_planner/fine_tuning/README.md):** generate
   fresh expert/LLM pairs, QP trajectories, position scores, and verification
   plots for either expert or a supplied dataset.

For a complete Semantic Theta example, follow
[generate 2,000 calibration rows](src/llm_vision_planner/fine_tuning/README.md#example-generate-2000-calibration-rows-in-the-expert-branch),
including the four generation/plot commands and output checks.

## 1. Clone, build, and source locally

Use Ubuntu 22.04 with ROS 2 Humble installed. Run dataset/calibration commands
with system Python, and GPU training/serving in their separate environments.

```bash
sudo apt update
sudo apt install -y git build-essential cmake python3-colcon-common-extensions \
  python3-rosdep python3-vcstool python3-pip python3-venv python3-numpy python3-scipy \
  python3-matplotlib python3-pytest curl
mkdir -p ~/Desktop
git clone --depth 1 --single-branch --branch finding_expert \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/starling_testing_ws
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
bash scripts/setup_workspace.sh
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update
rosdep install --from-paths src --ignore-src -r -y
/usr/bin/python3 -m pip install 'numpy<2' openai instructor pydantic
colcon build --symlink-install --packages-select px4_msgs llm_vision_planner
source install/setup.bash
```

Clone into an empty directory. If `main` already occupies this path, choose a
separate directory and substitute it throughout the local commands. Source this
clone's install in every new dataset-generation shell. DeltaAI training uses the
Slurm scripts in the NCSA guide and does not require a ROS installation.

## 2. Generate the expert instruction dataset

Both commands work without a model server. The CSVs retain the existing
`messages` schema consumed by both training scripts. These commands overwrite
their output CSVs. The existing expert datasets are retained for immediate use.

### RRT

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
cd src/llm_vision_planner
/usr/bin/python3 fine_tuning/scripts/conformal_rrt_dataset.py \
  --rrt-training --samples 20000 --random-goal --seed 7 \
  --output fine_tuning/datasets/rrt_expert_dataset.csv
```

### Semantic Theta*

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
cd src/llm_vision_planner
/usr/bin/python3 fine_tuning/scripts/conformal_semantic_theta_dataset.py \
  --semantic-theta-training --samples 20000 --random-goal --seed 17 \
  --output fine_tuning/datasets/semantic_theta_expert_dataset.csv
```

Semantic Theta* performs deterministic any-angle search with per-label hard
margins and soft traversal costs. Its dataset stores the policy and expert cost.
`conforml_semantic_theta_dataset.py` remains the existing compatibility spelling.
For a small check use `--samples 20` and a separate `--output` CSV. Keep training
seeds separate from the fresh calibration seed in the calibration README.

## 3. Fine-tune, download, and serve the selected adapter

Follow [NCSA_COMMANDS.md](NCSA_COMMANDS.md) for the matching expert. Both
`train_peft.py` and `train.py` remain available, with the four existing expert
Slurm entry points. The jobs use the original `bhkj-dtai-gh` allocation and
`/projects/bhkj/$USER/starling_testing_ws` path; change the account/path in the
existing scripts if your allocation differs.

No weights are checked in. Each `fine_tuning/outputs/llama31_8b_*_lora` directory
contains a clearly marked `PLACEHOLDER.txt`. Download/extract the real adapter
there before serving; placeholders cannot be used for inference.

On your GPU server, activate its vLLM environment. If needed, create one with
`python3 -m venv ~/vllm_env`, activate it, install `vllm`, and run `hf auth login`
using an account with gated Llama access. Choose one configuration below.

### Serve RRT

```bash
cd ~/Desktop/starling_testing_ws
ADAPTER="$PWD/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_rrt_lora"
test -s "$ADAPTER/adapter_config.json" && test -s "$ADAPTER/adapter_model.safetensors"
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-lora --max-lora-rank 128 \
  --lora-modules rrt_planner="$ADAPTER" --served-model-name rrt_planner \
  --dtype float16 --gpu-memory-utilization 0.80 --max-model-len 4096 --port 8000
```

### Serve Semantic Theta*

```bash
cd ~/Desktop/starling_testing_ws
ADAPTER="$PWD/src/llm_vision_planner/fine_tuning/outputs/llama31_8b_semantic_theta_lora"
test -s "$ADAPTER/adapter_config.json" && test -s "$ADAPTER/adapter_model.safetensors"
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-lora --max-lora-rank 128 \
  --lora-modules semantic_theta_planner="$ADAPTER" --served-model-name semantic_theta_planner \
  --dtype float16 --gpu-memory-utilization 0.80 --max-model-len 4096 --port 8000
```

The adapter checks must pass before running vLLM. Substitute the GPU clone's
actual path if different. Run one server command at a time on port 8000.
On the machine generating calibration pairs, check its reachable URL:

```bash
export VLLM_BASE_URL=http://172.22.224.93:8000/v1
curl --fail --silent --show-error "$VLLM_BASE_URL/models"
```

Replace the lab address with your GPU server. The selected model alias must
appear in `/models`, and must match `--llama-model-name` in the calibration
commands. Continue with the [calibration README](src/llm_vision_planner/fine_tuning/README.md).

## 4. Verify the offline pipeline

```bash
cd ~/Desktop/starling_testing_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
MPLBACKEND=Agg colcon test --packages-select llm_vision_planner
colcon test-result --verbose
```

The existing test exercises both expert schemas through QP generation, scoring,
and PNG/JSON verification, including an off-path sample that must fail. It does
not contact vLLM, submit training, or require hardware. Semantic calibration
CSVs contain headers only until generated; a plot test with synthetic pairs
does not turn them into measured calibration data.

## Working on different branches

`main` retains the full original workspace. Hardware reproduction and actual
Vicon vision-error collection are documented on
[`reporoduce_hardeware`](https://github.com/prachitgupta/starling_testing_ws/tree/reporoduce_hardeware).
Semantic Theta selection here applies to offline calibration/verification;
there is no Semantic Theta hardware launch in this branch.

Prefer separate clones so build/install products cannot mix:

```bash
git clone --depth 1 --single-branch --branch finding_expert \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/expert_ws
git clone --depth 1 --single-branch --branch reporoduce_hardeware \
  https://github.com/prachitgupta/starling_testing_ws.git ~/Desktop/hardware_ws
git -C ~/Desktop/expert_ws branch --show-current
git -C ~/Desktop/expert_ws pull --ff-only
```

Replace `~/Desktop/starling_testing_ws` with the appropriate clone path in local
commands, build there, and source only that clone in a fresh shell. Keep the
documented project path for NCSA jobs unless you also edit their `cd` commands.
Copy completed adapter/calibration artifacts between clones when needed; using
a branch does not require merging its file removals into `main`.

## Software validation of this branch

Verified with a fresh pinned PX4 message import and clean Humble build. The
existing test passes both expert schemas through QP generation, position
scoring, and PNG/JSON verification. Twenty-row training-data smoke runs pass
for each expert. Plotting retained RRT row 998 gives `s_p=0.202469 m` and
`q_p=0.235845 m`. Slurm shell syntax and the RRT job's final argument list were
checked; GPU training, live vLLM calibration, and hardware flight were not run.
