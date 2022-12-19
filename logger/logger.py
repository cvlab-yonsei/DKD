import logging
import logging.config
from pathlib import Path
from utils.utils import read_json


class Logger:
    def __init__(
        self,
        logdir,
        rank,
    ):
        self.rank = rank
        self.logger = None

        setup_logging(logdir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def set_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_levels[verbosity])

    # Print from all rank
    def print(self, msg):
        self.logger.info(msg)

    # Print from rank0 process only
    def info(self, msg):
        if self.rank == 0:
            self.logger.info(msg)

    def debug(self, msg):
        if self.rank == 0 and self.debug_flag:
            self.logger.info(msg)

    def error(self, msg):
        if self.rank == 0:
            self.logger.error(msg)

    def warning(self, msg):
        if self.rank == 0:
            self.logger.warning(msg)


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])
                
        logging.config.dictConfig(config)

    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
