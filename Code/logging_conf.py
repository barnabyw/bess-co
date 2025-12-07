import logging
import os

def setup_logging(output_path):
    log_file = os.path.join(output_path, "run.log")

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)   # ⬅️ Root level = WARNING only

    # Clear previous handlers
    logger.handlers.clear()

    # -------------------------------------------------
    # FILE HANDLER — log ONLY WARNING, ERROR, CRITICAL
    # -------------------------------------------------
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.WARNING)       # ⬅️ No INFO, no DEBUG in file log
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    ))
    logger.addHandler(fh)

    # -------------------------------------------------
    # CONSOLE — INFO-only
    # -------------------------------------------------
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda r: r.levelno == logging.INFO)
    info_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(info_handler)

    # -------------------------------------------------
    # CONSOLE — ERROR+
    # -------------------------------------------------
    error_handler = logging.StreamHandler()
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(error_handler)
