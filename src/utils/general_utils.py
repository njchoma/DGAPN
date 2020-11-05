import os
import wget
import logging

import torch


def initialize_logger(artifact_path, name=None, level='INFO'):
    logfile = os.path.join(artifact_path, 'log.txt')
    if name is None:
        logger = logging.getLogger()
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)

    handler_console = logging.StreamHandler()
    handler_file = logging.FileHandler(logfile)

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


def load_surrogate_model(artifact_path, surrogate_model_url, surrogate_model_path, device):
    if surrogate_model_url != '':
        surrogate_model_path = os.path.join(artifact_path, 'surrogate_model.pth')

        maybe_download_file(surrogate_model_path,
                            surrogate_model_url,
                            'Surrogate model')
    surrogate_model = torch.load(surrogate_model_path, map_location=device)
    print("Surrogate model loaded")
    return surrogate_model
