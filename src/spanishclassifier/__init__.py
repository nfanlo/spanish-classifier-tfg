import logging
import os
from logging.config import fileConfig
from typing import List

import coloredlogs
import pkg_resources  # type: ignore

__all__: List[str] = []
__copyright__: str = "Copyright 2023, Francisco Perez Sorrosal."
# __version__: str = pkg_resources.get_distribution("spanishclassifier").version

env_key = "LOG_CFG"
config_path = os.getenv(env_key, None)

env_key = "LOG_DEST"
home_dir = os.getenv("HOME", "/var/log")
log_dir = os.getenv(env_key, home_dir)
logfilename_path = log_dir + "/spanishclassifier.log"

if config_path is None or not os.path.exists(config_path):
    print("Basic logging config")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    coloredlogs.install(logger=logger)
else:
    print(f"Loading logging config from file {config_path}. Output to: {logfilename_path}")
    if not os.path.exists(log_dir):
        print(f"Creating dir {log_dir} for placing the log as it does not exist yet")
        os.makedirs(log_dir)
    fileConfig(
        config_path,
        defaults={"logfilename": logfilename_path},
        disable_existing_loggers=False,
    )
    logger = logging.getLogger()
