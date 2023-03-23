from time import localtime
import os

def get_folder(path):
    tm = localtime()
    path = path + f"{tm.tm_year}_{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}_{tm.tm_min}_{tm.tm_sec}"
    os.makedirs(path, exist_ok=True)
    
    return path