import time
from functools import wraps
from logging import INFO

from segmenter_api.utils.logger import get_logger

logger = get_logger(__name__)
STOPWATCH_LEVEL = INFO


def stop_watch(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start: float = time.time()
        start_msg = f"関数{f.__qualname__}が開始"
        logger.log(level=STOPWATCH_LEVEL, msg=start_msg)
        result = f(*args, **kwargs)
        elapsed_time: float = time.time() - start
        end_msg = f"関数{f.__qualname__}は{elapsed_time:.4f}secで処理を完了"
        logger.log(level=STOPWATCH_LEVEL, msg=end_msg)
        return result

    return wrapper
