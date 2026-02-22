import logging
import logging.config
import json
import traceback
import contextvars
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.core.settings import Settings

_CONFIGURED = False

# Context variable to store request ID across async tasks
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    """Filter that adds request ID to log records."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for production logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func_name": record.funcName,
            "line_number": record.lineno,
        }
        
        req_id = getattr(record, "request_id", "")
        if req_id:
            log_data["request_id"] = req_id
            
        if record.exc_info:
            log_data["exception"] = "".join(traceback.format_exception(*record.exc_info))
            
        # Add any extra attributes passed to log.info(..., extra={"key": "value"})
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
                           'funcName', 'levelname', 'levelno', 'lineno', 'message', 'module',
                           'msecs', 'msg', 'name', 'pathname', 'process', 'processName',
                           'relativeCreated', 'stack_info', 'thread', 'threadName', 'request_id']:
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except TypeError:
                    log_data[key] = str(value)

        return json.dumps(log_data)

def configure_logging(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = logging.INFO if settings.env != "dev" else logging.DEBUG
    
    log_format = (
        "%(asctime)s [%(levelname)s] [req_id:%(request_id)s] %(name)s:%(module)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_id": {
                "()": RequestIdFilter,
            },
        },
        "formatters": {
            "standard": {
                "format": log_format,
            },
            "json": {
                "()": JSONFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json" if settings.env != "dev" else "standard",
                "filters": ["request_id"],
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # Root logger
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "fastapi": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "app": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
        },
    }
    
    logging.config.dictConfig(logging_config)
    _CONFIGURED = True

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "app")
