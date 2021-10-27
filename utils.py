import logging

def log(log_file):
    """Returns a logger object with predefined settings"""
    root_logger = logging.getLogger(__name__)
    root_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    return root_logger