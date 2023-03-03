import psutil
from datetime import datetime
from time import sleep
import logging

logging.basicConfig(filename="delucs.log",
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

def get_current_time() -> str:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

while True:
    current_time = get_current_time()
    current_cpu_util = psutil.cpu_percent()
    current_memory = psutil.virtual_memory()

    logging.info(f"Current time: {current_time}")
    logging.info(f"Current CPU Util :{current_cpu_util}")
    logging.info(f"Current Memory:\n {current_memory} \n")
    sleep(10)