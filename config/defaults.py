'''Configs'''

from fvcore.common.config import CfgNode


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

_C.DATA.NUM_WORKERS = 0

_C.DATA.PIN_MEMORY = True

_C.DATA.ROOT = './.data'


# -----------------------------------------------------------------------------
# Training options
# -----------------------------------------------------------------------------
_C.TRAIN = CfgNode()

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 128

# Maximal number of epochs.
_C.TRAIN.MAX_EPOCH = 300

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = './cache/res18.conv2d.165.0220.pth'

# Checkpoint dir.
_C.TRAIN.CHECKPOINT_DIR = './checkpoint'


# -----------------------------------------------------------------------------
# Optimizer options
# -----------------------------------------------------------------------------
_C.SOLVER = CfgNode()

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = 'sgd'

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Weight decay value that applies on BN.
_C.SOLVER.BN_WEIGHT_DECAY = 0.0

# Gradually warm up the TRAIN.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Learning rate policy (see utils/optim_util.py for options and examples).
_C.SOLVER.LR_POLICY = 'cosine'

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []


# -----------------------------------------------------------------------------
# Misc options
# -----------------------------------------------------------------------------
# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1


def get_cfg():
    return _C.clone()
