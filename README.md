# Training LLM with MS-Swift on ROCm

## Quick Start

### Preparation
```bash
git clone --recursive https://github.com/Treemann/ms-swift-training-with-rocm.git
cd ms-swift-training-with-rocm

# Prepare the model and dataset of interest
export MODELSCOPE_CACHE=${PWD}/shared
modelscope download --model Qwen/Qwen3-8B
modelscope download --model Qwen/Qwen3-32B
modelscope download --dataset swift/chinese-c4
```

### Single-node Training
```bash
bash run_1node_8b.sh
bash run_1node_32b.sh
```

### Multi-node Training
- Step1: Follow the instruction in `tools/use_host_bcm_ib_driver_in_container.sh` to properly set up the InfiniteBand driver (script `launch_rocm_container.sh` is for launch the container).
- step2: Refer to `run_2node_ranks.sh` for running multi-node training.

- Qwen3-8B
```
# master node
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.2.96.7 IP_INTERFACE=enp49s0f1np1 MASTER_PORT=1978 bash run_multinodes_8b.sh
# worker node
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.2.96.7 IP_INTERFACE=enp49s0f1np1 MASTER_PORT=1978 bash run_multinodes_8b.sh
```

- Qwen3-32B
```
# master node
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.2.96.7 IP_INTERFACE=enp49s0f1np1 MASTER_PORT=1978 bash run_multinodes_32b.sh
# worker node
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.2.96.7 IP_INTERFACE=enp49s0f1np1 MASTER_PORT=1978 bash run_multinodes_32b.sh
```