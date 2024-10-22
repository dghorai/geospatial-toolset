import os
import logging

from datetime import datetime


# create log folder
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FOLDER = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_FOLDER, exist_ok=True)

# create logfile
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_FOLDER, LOG_FILE_NAME)

# add basic config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
