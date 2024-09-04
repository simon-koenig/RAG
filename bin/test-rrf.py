# Test of rag evaluation
# Imports
import sys
from pprint import pprint

import openpyxl

sys.path.append("./dev/")
sys.path.append("./src/")
import csv

from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("minibios-qa-gpu")  # Connect to the minibio
stats = documentDB.getIndexStats()
print(stats)


query = "what is pharma"
num_ref = 2
lang = "EN"
rerank = "rrf"
documentDB.getBackground(
    query=query, num_ref=num_ref, lang=lang, rerank=rerank, prepost_context=False
)
