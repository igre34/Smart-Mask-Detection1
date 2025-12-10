import os 

# SYSTEM CONFIGURATIONS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
PROTOTXT_PATH = "ai/deploy.prototxt"
WEIGHTS_PATH = "ai/res10_300x300_ssd_iter_140000.caffemodel"
MODEL_PATH = "ai/mask_detector.h5"

# CONSTANTS DEFINITIONS
CONFIDENCE_THRESHOLD = 0.5
SNAPSHOT_COOLDOWN = 5

