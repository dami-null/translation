# 路径参数
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]

DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'
LOGS_DIR = ROOT_DIR / 'logs'

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30

# 模型参数
DIM_MODEL = 256
NUM_HEAD = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 4 * DIM_MODEL

# 生成参数
MAX_STEPS=100