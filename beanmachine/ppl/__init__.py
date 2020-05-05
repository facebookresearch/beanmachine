import logging


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
file_handler = logging.FileHandler("beanmachine.log")
file_handler.setLevel(logging.INFO)

LOGGER = logging.getLogger("beanmachine")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)
