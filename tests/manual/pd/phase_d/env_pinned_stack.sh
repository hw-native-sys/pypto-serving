#!/usr/bin/env bash
# Source this file to select the Phase D D0 frozen stack.
source /home/sj/git/env_all.sh

# The validation containers are not built from the same image.  Do not inherit
# image-level toolkit aliases; select one CANN root and one project stack.
unset ASCEND_AICPU_PATH
unset ASCEND_TOOLKIT_HOME
unset ASCEND_TOOLKIT_LATEST_HOME
unset TOOLCHAIN_HOME
unset CMAKE_PREFIX_PATH
unset PYTHONHOME
unset PYTHONSTARTUP
unset LD_PRELOAD

export DSPARK_STAGE
DSPARK_STAGE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYPTO_HOME="$DSPARK_STAGE/pypto"
export PYPTO_ROOT="$PYPTO_HOME"
export SIMPLER_ROOT="$PYPTO_HOME/runtime"
export PYPTO_LIB_ROOT="$DSPARK_STAGE/pypto-serving/pypto-lib"
export PYPTO_LIB_HOME="$PYPTO_LIB_ROOT"
export SIMPLER_BUILD_LIB_DIR="$SIMPLER_ROOT/build/lib"
export PYTHONPATH="$DSPARK_STAGE/pypto-serving:$PYPTO_HOME/python:$SIMPLER_ROOT:$SIMPLER_ROOT/python:$PYPTO_LIB_ROOT:$MLIR_PYTHON_ROOT"
export LD_LIBRARY_PATH="$SIMPLER_BUILD_LIB_DIR:$SIMPLER_BUILD_LIB_DIR/a2a3/dispatcher:$LLVM_BUILD_DIR/lib:$ASCEND_HOME_PATH/aarch64-linux/lib64:$ASCEND_HOME_PATH/runtime/lib64:$ASCEND_DRIVER_LIB_PATH:$MOONCAKE_LIB_DIR:$MOONCAKE_LIB64_DIR"

source "$SIMPLER_ROOT/.venv/bin/activate"
export PTOAS_ROOT="$VIRTUAL_ENV/bin"
export PTO_SOURCE_DIR="$DSPARK_STAGE/ptoas"
export PTOAS_SOURCE_DIR="$PTO_SOURCE_DIR"
export PTO_BUILD_DIR="$PTO_SOURCE_DIR/build"
export PTO_ISA_ROOT="$SIMPLER_ROOT/build/pto-isa"
export PTO_TILE_LIB_CODE_PATH="$PTO_ISA_ROOT"

# Keep executable selection deterministic.  Do not fall through to an
# image-provided PTOAS or an alternate project checkout.
export PATH="$VIRTUAL_ENV/bin:/usr/local/python3.11.14/bin:$ASCEND_HOME_PATH/bin:$ASCEND_HOME_PATH/tools/ccec_compiler/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin:/root/bin"
export PYTHONNOUSERSITE=1
export CANN_ROOT="$ASCEND_HOME_PATH"
export SOC_VERSION=ascend910_9391
export TASK_QUEUE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export ASCEND_PROCESS_LOG_PATH="$DSPARK_STAGE/ascend"

# Model runs set their ring sizes explicitly.  Do not inherit diagnostic values.
unset PTO2_RING_HEAP PTO2_RING_TASK_WINDOW PTO2_RING_DEP_POOL
