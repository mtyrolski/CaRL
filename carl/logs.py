from loguru import logger


def warn(msg: str) -> None:
    logger.warning(msg)


def info(msg: str) -> None:
    logger.info(msg)


def debug(msg: str) -> None:
    logger.debug(msg)


def success(msg: str) -> None:
    logger.success(msg)
