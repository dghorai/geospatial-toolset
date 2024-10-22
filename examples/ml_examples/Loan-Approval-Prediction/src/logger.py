import os
import sys
import logging

# get source folder
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# setup logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("LoanApprovalPredLogger")
