# Test of rag evaluation
# Imports
import logging
import os
import sys
import time
from functools import partial

import numpy as np

sys.path.append("./dev/")
sys.path.append("./src/")

import concurrent.futures
import itertools

from dataset_helpers import DatasetHelpers
from drivers import pipe_single_setting_run
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "mixtral:latest"  # "llama3.1:latest",  "llama3.1:70b", "llama3-chatqa:8b", "mixtral:latest"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"

# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniWiki()
)  # datasetHelpers.loadMiniBiosqa() , datasetHelpers.loadMiniWiki()

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex(
    "miniwiki-gpu"
)  # [{'indexName': 'minibios-qa-gpu'}, {'indexName': 'miniwiki-gpu'}]]
stats = documentDB.getIndexStats()
print(stats)

##
## Set parameters for the pipeline
##

query_expansion = [1, 2, 3]
rerank = [
    "rrf",
    True,
]  #  "rrf", False]    # , "rrf", False]  # True, "rrf", False
prepost_context = [False]
background_reversed = [False]
num_ref_lim = [5]  # [1, 2, 4, 6] [5]

##
## Calculate the dimensionality of the parameter space
##

parameter_count = np.prod(
    [
        len(query_expansion),
        len(rerank),
        len(prepost_context),
        len(background_reversed),
        len(num_ref_lim),
    ]
)

##
## Run pipeline for all parameter permutations. in parallel
##


# Define parameter space
parameters = list(
    itertools.product(
        query_expansion, rerank, prepost_context, background_reversed, num_ref_lim
    )
)


# Run pipeline for all parameter permutations
n_sample_queries = 100
# Manually adjust this path to the desired output directory with the variable parameter
write_to_dir = "./parallel_100_rows_pipe/miniWiki/quExp/"
# make sure to create the directory before running the script
os.makedirs(write_to_dir, exist_ok=True)

# Define helper function
dev_helper_func = partial(
    pipe_single_setting_run,
    queries=queries,
    ground_truths=ground_truths,
    goldPassages=goldPassages,
    documentDB=documentDB,
    LLM_URL=LLM_URL,
    LLM_NAME=LLM_NAME,
    n_sample_queries=n_sample_queries,
    write_to_dir=write_to_dir,
)
# Make test run and time for n workers
# Run the pipeline in parallel


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Parallel
# start = time.time()
# with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
#     futures = [executor.submit(dev_helper_func, param) for param in parameters]
#     for future in concurrent.futures.as_completed(futures):
#         try:
#             future.result()
#         except Exception as exc:
#             logging.error(f"Generated an exception: {exc}")
#     print(f"Execution with {n_worker} workers")


# end = time.time()
# temp = end - start
# outfile.write(f"Execution with {n_worker} workers took {temp} seconds\n")


# Sequential with parallel under the hood
start = time.time()
for param in parameters:
    dev_helper_func(param)
end = time.time()
temp = end - start

print(f"Execution time: {temp}")
print("Parameter count: ", parameter_count)
print("Execution time per parameter setting: ", temp / parameter_count)
