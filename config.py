# Import from config file
import configparser

# Initialize the parser and read the config file
config = configparser.ConfigParser()
config.read("config.ini")

# Define API endpoints
LLM_URL = config["API"]["LLM_URL"]
MARQO_URL = config["API"]["MARQO_URL"]
MARQO_URL_GPU = config["API"]["MARQO_URL_GPU"]
RERANKER_ENDPOINT = config["API"]["RERANKER_ENDPOINT"]
