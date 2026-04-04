"""
Top-level package for geoutils-rsg.
"""

__version__ = '0.0.3'
__author__ = 'Dr. Debabrata Ghorai'
__email__ = 'ghoraideb@gmail.com'

import os
import sys
import logging
import datetime


class CustomException(Exception):

    def __init__(self, err, sys_err: sys):
        super().__init__(err)
        self.err_msg = error_handling(err, sys_err=sys_err)

    def __str__(self):
        return self.err_msg


def error_handling(err, sys_err: sys):
    # using the sys module to access more of the exception's information with sys.exc_info
    _, _, exp_trace_back = sys_err.exc_info()
    file_name = exp_trace_back.tb_frame.f_code.co_filename
    err_msg = f"Error occured in python script name [{file_name}] line number [{exp_trace_back.tb_lineno}] error message [{str(err)}]"
    return err_msg


# create log folder and file
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(project_dir, "logs")
# log_file_name = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# setup logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

# add basic config
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("RSGToolsLogger")
