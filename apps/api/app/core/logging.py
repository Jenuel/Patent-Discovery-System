from __future__ import annotations

import logging
from typing import Optional

from app.core.settings import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = logging.INFO if settings.env != "dev" else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "app")
