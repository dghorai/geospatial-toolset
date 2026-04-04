import os
import sys
import logging
import functools


# setup logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"

# create log folder
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok=True)

# add basic config
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GeospatialLogger")


class CustomException(Exception):
    """
    Raise custom exception in any block of codes.
    """

    def __init__(self, message, sys_err: sys):
        self.message = message
        super().__init__(message)
        self.message = error_handling(message, sys_err=sys_err)

    def __str__(self):
        return self.message
    

def error_handling(err, sys_err: sys):
    # using the sys module to access more of the exception's information with sys.exc_info
    _, _, exp_trace_back = sys_err.exc_info()
    file_name = exp_trace_back.tb_frame.f_code.co_filename
    err_msg = f"Error occured in python script name [{file_name}] line number [{exp_trace_back.tb_lineno}] error message [{str(err)}]"
    return err_msg


# define custom exception function
def handle_exceptions(function):
    """
    General decorator to wrap try-except in python.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__
            logging.exception(err)
            # re-raise the exception
            raise
    return wrapper
