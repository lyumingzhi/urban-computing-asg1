import logging
import os
import shutil
import time


def get_logger(path):
    logger = logging.getLogger('UrbanComputing-A1')
    logger.setLevel(logging.INFO)
    msg = []
    if not os.path.isdir(path):
        msg.append(f'{path} not exist, make it')
        os.mkdir(path)
    log_file_path = os.path.join(path, 'log.log')
    if os.path.isfile(log_file_path):
        target_path = f'{log_file_path}.{time.strftime("%Y%m%d%H%M%S")}'
        msg.append(f'Log file exists, backup to {target_path}')
        shutil.move(log_file_path, target_path)
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
