import logging

logFormatter = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logging.getLogger(__name__).addHandler(logging.NullHandler())
