import os
import wget
import logging

def initialize_logger(artifact_path, name=None, level='INFO'):
    logfile = os.path.join(artifact_path, 'log.txt')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    handler_console = logging.StreamHandler()
    handler_file    = logging.FileHandler(logfile)

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)
    return logger

def close_logger(logger=None):
    if logger is None:
        logger = logging.getLogger()
    handlers = logger.handlers
    for h in handlers:
        h.close()
    for i in range(len(logger.handlers)):
        logger.handlers.pop()

def maybe_download_file(file_path, url, file_descriptor):
    print()
    if not os.path.isfile(file_path):
        print("{} not found. Downloading.".format(file_descriptor))
        wget.download(url, file_path)
    else:
        print("{} found.".format(file_descriptor))
    print()
