import os
import toml
from pathlib import Path
from dacite import from_dict
from .config import Config

# Resolve default configuration folder
HOME_PATH = os.environ["HOME"]

if "SEMS_CONFIG_FOLDER" in os.environ:
    CONFIG_PATH = os.environ["SEMS_CONFIG_FOLDER"]
else:
    CONFIG_PATH = os.environ["XDG_CONFIG_HOME"] \
        if "XDG_CONFIG_HOME" in os.environ \
        else os.path.join(HOME_PATH,
                          '.config')
    CONFIG_PATH = os.path.join(CONFIG_PATH, 'sems-processors')

CONFIG_FILE = os.path.join(CONFIG_PATH, "config.toml")

DEFAULT_CONFIG = {
    'camera_service': {
        'hostname': 'localhost',
        'port': 3000,
        'use_https': False,
    },
    'camera_server': {
        'hostname': 'localhost',
        'port': 3001,
        'use_https': False,
    },
}




def load_config(logger) -> Config:
    """
    Loads the configuration file as defined by the SEMS_CONFIG_FOLDER and XDG_CONFIG_HOME environment variables.
    If neither are available, the default $HOME/.config is selected.
    If the configuration file does not exist yet, it's created with the default configuration defined above.
    @return: Configuration mapping
    """
    config_folder = Path(CONFIG_PATH)
    if not config_folder.exists() or not config_folder.is_dir():
        os.makedirs(config_folder)

    config_file = Path(CONFIG_FILE)

    if not config_file.is_file():
        logger.info("configuration file does not exist, creating with default values")
        with open(config_file, 'w') as cf:
            toml.dump(DEFAULT_CONFIG, cf)

    with open(config_file, 'r') as cf:
        logger.info("loaded configuration file from %s", config_file)
        config = toml.load(cf)

    return from_dict(Config, config)
