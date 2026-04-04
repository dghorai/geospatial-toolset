import sys


def error_handling(err, sys_err: sys):
    # using the sys module to access more of the exception's information with sys.exc_info
    _, _, exp_trace_back = sys_err.exc_info()
    file_name = exp_trace_back.tb_frame.f_code.co_filename
    err_msg = f"Error occured in python script name [{file_name}] line number [{exp_trace_back.tb_lineno}] error message [{str(err)}]"
    return err_msg


class CustomException(Exception):
    
    def __init__(self, err, sys_err: sys):
        super().__init__(err)
        self.err_msg = error_handling(err, sys_err=sys_err)
    
    def __str__(self):
        return self.err_msg
