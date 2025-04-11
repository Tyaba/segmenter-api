import json
import logging
import sys
import traceback
from datetime import datetime, timedelta, timezone
from logging import Logger
from pathlib import Path

DATEFMT = "%Y-%m-%d_%H:%M:%S"
JST = timezone(timedelta(hours=+9), "JST")
NOW = datetime.now(tz=JST).strftime(DATEFMT)
LOG_DIR = Path("log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILENAME = LOG_DIR / f"{NOW}.log"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record=record)
        body = {
            "severity": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "message": message,
        }

        # traceback
        if record.exc_info:
            body["traceback"] = traceback.format_exc()

        return json.dumps(body, ensure_ascii=False)


stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(JsonFormatter(datefmt=DATEFMT))
file_handler = logging.FileHandler(filename=LOG_FILENAME)
file_handler.setFormatter(JsonFormatter(datefmt=DATEFMT))


def get_logger(
    name: str,
    stream_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    # loggerをinit
    logger = logging.getLogger(name)
    # level指定はhandlerで行う。根本のloggerはDEBUGまで全て通す
    logger.setLevel(logging.DEBUG)
    logger = attach_handler(
        logger=logger, stream_level=stream_level, file_level=file_level
    )
    return logger


def attach_handler(
    logger: Logger,
    stream_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> Logger:
    # streamに流すログを設定
    stream_handler.setLevel(stream_level)
    logger.addHandler(stream_handler)
    # fileに流すログを設定
    file_handler.setLevel(file_level)
    logger.addHandler(file_handler)
    return logger
