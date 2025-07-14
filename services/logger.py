import logging
from rich.logging import RichHandler

def get_logger(name: str = "askbit", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(level)
        handler = RichHandler(rich_tracebacks=True, markup=True)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
