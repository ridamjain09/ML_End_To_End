''''
check the documentation 
we use this to log the exceptions and everything

'''

import logging
import logging.config
import os
from datetime import datetime

#Log file will be created in this naming convention 
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),'logs',LOG_FILE) #Path of log file
#Keep on appending Files when also files exists
os.makedirs(log_path, exist_ok=True)
#Log Files path 
LOG_FILE_PATH = os.path.join(log_path,LOG_FILE)


logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, #In case of logging will print this file path 
    )

