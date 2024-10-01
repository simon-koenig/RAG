# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("miniwiki-gpu")  # Connect to the minibio
stats = documentDB.getIndexStats()
print(stats)


query = "Who does Avogadro know?"
# # Print results
# for elem in pipe.rag_elements:
#    pprint(elem)

background, contexts, context_ids = documentDB.getBackground(query, rerank=False)
pprint(contexts)
pprint(context_ids)
