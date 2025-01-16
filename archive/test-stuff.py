# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
sys.path.append("./")

from config import LLM_URL, MARQO_URL, MARQO_URL_GPU
from dataset_helpers import DatasetHelpers
from vector_store import VectorStore

# Define LLM Name if necessary
LLM_NAME = "llama3.1:latest"

datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # Mini Wiki

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("minibios-qa-gpu")  # Connect to the minibio


# # Print results
# for elem in pipe.rag_elements:
#    pprint(elem)
query = queries[1]
goldPassages = goldPassages[1]
background, contexts, context_ids = documentDB.getBackground(
    query,
    rerank=True,
    query_expansion=3,
    LLM_NAME=LLM_NAME,
    LLM_URL=LLM_URL,
    search_ref_lex=4,
    search_ref_sem=4,
    num_ref_lim=5,
)
pprint(f"Query: {query}")
pprint(f"Context IDs: {context_ids}")
pprint(f"Gold Passages: {goldPassages}")
# unique matches of gold passages in context_ids
matches = [1 for p in goldPassages if p in context_ids]
print("matches: ", matches)
print("matches: ", len(matches))


# Load the RagPipe
# pipe = RagPipe()
# pipe.connectVectorStore(documentDB)
# pipe.connectLLM(LLM_URL, LLM_NAME)
# pipe.answerQuery(query)
