# Test of rag evaluation
# Imports
import sys
import time
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
from csv_write import write_context_relevance_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = datasetHelpers.loadMiniBioasq()


# Load the VectorStore on marqo 2.10 with gpu
documentDB_GPU = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
documentDB_GPU.connectIndex("minibios-qa-gpu")
print(documentDB_GPU.getIndexes())  # Print all indexes

##
## Index Docs and measure time
##

# maxDocs = 43000


# Test index of maxDocs documents
# start = time.time()
# documentDB_GPU.indexDocuments(corpus_list, maxDocs)  # Add documents to the index
# end = time.time()
# ## Time for indexing 100 documents on marqo with cpu
# print(f"Time for indexing {maxDocs} documents: {end - start} seconds")


# documentDB_GPU.emptyIndex()  # Empty the index
print(documentDB_GPU.getIndexStats())  # Print index stats
