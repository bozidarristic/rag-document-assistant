import logging


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, # INFO, WARNING, ERROR
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)