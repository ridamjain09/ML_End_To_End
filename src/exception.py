'''
whenever you want to write a exception 
we can search for exception Python on google 
'''

import sys  #Use to manipulate diffrent parts of Python runtime enviorment 
import logging
from src.logger import logging


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() #this will give us the exception information(file,line_number all the details)
   #How to get file name 
    file_name = exc_tb.tb_frame.f_code.co_filename
    #Place holder  while giving error message 
    error_message = "Error Occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)
    
    #Return Error message in string 
    def __str__(self):
        return self.error_message
    
