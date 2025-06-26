import logging
from contextvars import ContextVar
from logging import LogRecord
from logging.handlers import RotatingFileHandler

request_id: ContextVar[str] = ContextVar("request_id", default="N/A")


class ContextLogRecord(LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_id = request_id.get("N/A")  # Use "N/A" if not set


# Update the Logger to use this custom record
logging.setLogRecordFactory(ContextLogRecord)

# Define the logger
logger = logging.getLogger("manga_frontend-logs")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # Disable propagation to parent loggers

# Check if handlers already exist before adding new ones
if not logger.handlers:
    # Create handlers with different levels
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)

    f_handler = RotatingFileHandler("manga_frontend.log", maxBytes=10000, backupCount=0, encoding="utf-8")
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add them to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - [Request ID: %(request_id)s] - %(message)s")
    f_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [Request ID: %(request_id)s] - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
