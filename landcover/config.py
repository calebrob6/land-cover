# Configuration of RUN.PY
VERBOSE = 2
## Directories
DATA_DIR = "chesapeake_data/"
OUTPUT_DIR = "results/"

## States
TRAINING_STATES = ["ny_1m_2013"]
VALIDATION_STATES = ["ny_1m_2013"]
TEST_STATES = ["ny_1m_2013"]
### Only used if loss == superres
SUPERRES_STATES = []

## Augmentation
DO_COLOR = False

## Model settings
MODEL = "unet"

## Training settings
EPOCHS = 100
LOSS = "crossentropy"
LEARNING_RATE = 0.001
BATCH_SIZE = 16  # use 128 for full dataset (this is for NY only)

# Other settings
# NOTE: If you want to remove var, assign to None
## Number of target classes
HR_NCLASSES = 5
LR_NCLASSES = 22

## Positional index of labels in patches
HR_LABEL_INDEX = 8
LR_LABEL_INDEX = 9

## Datatype of imagery
# should be int8 (divide by 255) or int16 (divide by 10000)
DATA_TYPE = "int8"

## Keys for transformation of labels
HR_LABEL_KEY = "data/cheaseapeake_to_hr_labels.txt"
LR_LABEL_KEY = "data/nlcd_to_lr_labels.txt"

## COLORMAP files for labels
HR_COLOR = "data/hr_color.txt"
LR_COLOR = "data/nlcd_color.txt"

## LR files used for superres loss
LR_STATS_MU = "data/nlcd_mu.txt"
LR_STATS_SIGMA = "data/nlcd_sigma.txt"
LR_CLASS_WEIGHTS = "data/nlcd_class_weights.txt"

# Weights to use in transfer learning
# Will load file if not empty
# MAKE SURE ARCHITECTURE IS THE SAME
PRELOAD_WEIGHTS = ""
