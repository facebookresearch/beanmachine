import logging

from beanmachine.ppl.model import functional, random_variable


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
file_handler = logging.FileHandler("beanmachine.log")
file_handler.setLevel(logging.INFO)

LOGGER = logging.getLogger("beanmachine")
LOGGER.setLevel(logging.INFO)
LOGGER.handlers.clear()
LOGGER.addHandler(console_handler)
LOGGER.addHandler(file_handler)

__all__ = ["functional", "random_variable"]
