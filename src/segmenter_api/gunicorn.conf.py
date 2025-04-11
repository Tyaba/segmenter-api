from __future__ import annotations

import os
from typing import Any

wsgi_app: str = "src.segmenter_api.app.server:api"
bind: list[str] = [f"0.0.0.0:{int(os.getenv('PORT', '8080'))}"]
workers: int = 1
worker_class: str = "uvicorn.workers.UvicornWorker"
timeout: int = 0
reload: bool = True if os.environ.get("DEBUGGING") else False
logconfig_dict: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"json": {"()": "src.segmenter_api.utils.logger.JsonFormatter"}},
    "handlers": {
        "default": {
            "formatter": "json",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "root": {"level": "WARNING", "handlers": []},
    "loggers": {
        "gunicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "gunicorn.error": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": False,
        },
        "gunicorn.access": {
            "handlers": ["default"],
            "level": "WARNING",
            "propagate": False,
        },
    },
}
