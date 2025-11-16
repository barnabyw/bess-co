import logging
import os

def setup_logging(output_path):
    log_file = os.path.join(output_path, "run.log")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear previous handlers
    logger.handlers.clear()

    # -----------------------
    # File handler (everything)
    # -----------------------
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    ))
    logger.addHandler(fh)

    # -----------------------
    # Console INFO-only handler
    # -----------------------
    info_handler = logging.StreamHandler()
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(lambda r: r.levelno == logging.INFO)
    info_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(info_handler)

    # -----------------------
    # Console ERROR+ handler (ERROR & CRITICAL)
    # -----------------------
    error_handler = logging.StreamHandler()
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logger.addHandler(error_handler)
