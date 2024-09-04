# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
import csv

from csv_helpers import read_pipe_results_from_csv, write_pipe_results_to_csv
from dataset_helpers import DatasetHelpers
from evaluate import eval
from pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"

# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # Mini Bios

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex("minibios-qa-gpu")  # Connect to the minibio
stats = documentDB.getIndexStats()
print(stats)

# Load the RagPipe
pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)


# Run pipeline
pipe.run(
    questions=queries[:1],
    ground_truths=ground_truths,
    goldPassagesIds=goldPassages,
    corpus_list=corpus_list,
    newIngest=False,
    maxDocs=100000,
    maxQueries=1000000,
    lang="EN",
    query_expansion=3,
    rerank=True,
    prepost_context=False,
    background_reversed=True,
)

# Print results
for elem in pipe.rag_elements:
    pprint(elem)


write_pipe_results_to_csv(pipe.rag_elements, "./rag_results/query_expansion_test.csv")
print("Results written to csv file.")
