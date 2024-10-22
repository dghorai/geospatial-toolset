import os

from pathlib import Path
from config import PRJ_DIR

CONFIG_FILE_PATH = Path(os.path.join(PRJ_DIR, "src", "feature_selection", "config.yaml"))
PARAMS_FILE_PATH = Path(os.path.join(PRJ_DIR, "src", "feature_selection", "params.yaml"))
