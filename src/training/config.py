MODEL_NAME = "vinai/phobert-base"

TRAIN_PATH = "dataset/processed/train.csv"

VAL_PATH = "dataset/processed/val.csv"

TEST_PATH = "dataset/processed/test.csv"

OUTPUT_DIR = "checkpoints/phobert_bias_classifier"

MAX_LENGTH = 128

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

EPOCHS = 5

NUM_LABELS = 13

SEED = 42