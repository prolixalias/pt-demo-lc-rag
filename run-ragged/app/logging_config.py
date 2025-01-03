import logging
import logging.config # TODO: is this necessary?
import os
import sys

##############################
#
# DEBUG 	logging.debug()
# INFO 	    logging.info()
# WARNING 	logging.warning()
# ERROR 	logging.error()
# CRITICAL 	logging.critical()
#
##############################

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console"],
    },
}

print(f"Full Logging Configuration: {LOGGING_CONFIG}") # Log the full config

def setup_logging():
    """Sets up logging based on the shared configuration with environment variables."""
    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    # Ensure log level is one of the accepted logging levels
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        log_level = "DEBUG"  # Default to DEBUG if invalid
        print(f"Invalid LOG_LEVEL, defaulting to DEBUG")
    
    LOGGING_CONFIG["root"]["level"] = log_level
    logging.config.dictConfig(LOGGING_CONFIG)