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
LLM_NAME = "llama3.1:latest"
LLM_70B_NAME = "llama3.1:70b"
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


##
## Set parameters for the pipeline
##

query_expansion = [0, 1, 2, 3, 4, 5]

for query_expansion_val in query_expansion:
    pipe.setConfigs(
        lang="EN",
        query_expansion=query_expansion_val,
        rerank=True,
        prepost_context=False,
        background_reversed=False,
        search_ref_lex=2,
        search_ref_sem=2,
        num_ref_lim=4,
        model_temp=0.1,
        answer_token_num=100,
    )

    # Run pipeline
    pipe.run(
        questions=queries[:1],
        ground_truths=ground_truths[:1],
        goldPassagesIds=goldPassages[:1],
    )

    # Print results
    # for elem in pipe.rag_elements:
    #    pprint(elem)

    # Write results to csv file
    # Build the csv file path for the current parameter setting
    csv_file_path = f"./pipe_results/query_expansion{query_expansion_val}.csv"
    write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)
    print("Results written to csv file.")
