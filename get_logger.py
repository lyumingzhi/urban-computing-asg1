
import logging
import shutil
import time

def get_logger(path):
  logger = logging.getLogger('UrbanComputing-A1')
  logger.setLevel(logging.INFO)
  if not os.path.isdir(path):
    msg.append('%s not exist, make it' % path)
    os.mkdir(path)
  log_file_path = os.path.join(path, 'log.log')
  if os.path.isfile(log_file_path):
    target_path = log_file_path + '.%s' % time.strftime("%Y%m%d%H%M%S")
    msg.append('Log file exists, backup to %s' % target_path)
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
