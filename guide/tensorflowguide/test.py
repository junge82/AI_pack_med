
import tensorflow as tf
import logging as log
import sys

FILE_EXTENSIONS = [".nii.gz", ".tar.gz"]
CONSOLE_LOG_FORMAT = "\033[1m%(levelname)s:niftynet:\033[0m %(message)s"
FILE_LOG_FORMAT = "%(levelname)s:brats17:%(asctime)s: %(message)s"

def set_logger(file_name=None):
    """
    Writing logs to a file if file_name,
    the handler needs to be closed by `close_logger()` after use.

    :param file_name:
    :return:
    """
    # pylint: disable=no-name-in-module
    from tensorflow.python.platform.tf_logging import _get_logger

    logger = _get_logger()
    tf.logging.set_verbosity(tf.logging.INFO)
    logger.handlers = []

    # adding console output
    #f = log.Formatter(CONSOLE_LOG_FORMAT)
    std_handler = log.StreamHandler(sys.stdout)
    #std_handler.setFormatter(f)
    logger.addHandler(std_handler)

    if file_name:
        # adding file output
        f = log.Formatter(FILE_LOG_FORMAT)
        file_handler = log.FileHandler(file_name)
        file_handler.setFormatter(f)
        logger.addHandler(file_handler)


def close_logger():
    """
    Close file-based outputs

    :return:
    """
    # pylint: disable=no-name-in-module
    from tensorflow.python.platform.tf_logging import _get_logger

    logger = _get_logger()
    for handler in reversed(logger.handlers):
        try:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        except (OSError, ValueError):
            pass


set_logger('kecske.log')

tf.logging.info('1+1=2')