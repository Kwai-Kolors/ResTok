# Modified from:
#   TiTok: https://github.com/bytedance/1d-tokenizer/blob/main/utils/logger.py

import functools
import sys
from accelerate.logging import MultiProcessAdapter
import logging
from termcolor import colored

from iopath.common.file_io import PathManager as PathManagerClass

__all__ = ["setup_logger", "PathManager"]

PathManager = PathManagerClass()


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def setup_logger(name="ResTok", log_level: str = None, color=True, use_accelerate=True,
                 output_file=None):
    logger = logging.getLogger(name)
    if log_level is None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level.upper())

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_file is not None:
        fileHandler = logging.FileHandler(output_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    if use_accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger


# import logging
# import torch.distributed as dist


# def create_logger(logging_dir):
#     """
#     Create a logger that writes to a log file and stdout.
#     """
#     if dist.get_rank() == 0:  # real logger
#         logging.basicConfig(
#             level=logging.INFO,
#             format='[\033[34m%(asctime)s\033[0m] %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S',
#             handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
#         )
#         logger = logging.getLogger(__name__)
#     else:  # dummy logger (does nothing)
#         logger = logging.getLogger(__name__)
#         logger.addHandler(logging.NullHandler())
#     return logger