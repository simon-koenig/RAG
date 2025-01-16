# Test of rag evaluation
# Imports
import configparser
import logging
import os
import sys
import time
from functools import partial

sys.path.append("./dev/")
sys.path.append("./src/")
sys.path.append("./")


import itertools

from config import LLM_URL, MARQO_URL, MARQO_URL_GPU
from dataset_helpers import DatasetHelpers
from drivers import pipe_single_setting_run
from vector_store import VectorStore

print("Imports from config file:")
print("LLM_URL:", LLM_URL)
print("MARQO_URL:", MARQO_URL)
print("MARQO_URL_GPU:", MARQO_URL_GPU)

# Define LLM Name if necessary
# LLM_NAME = "llama3.1:latest"
# Load queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # datasetHelpers.loadMiniBiosqa() , datasetHelpers.loadMiniWiki()

# Load the VectorStore
documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
documentDB.connectIndex(
    "minibios-qa-gpu"
)  # [{'indexName': 'minibios-qa-gpu'}, {'indexName': 'miniwiki-gpu'}]]
stats = documentDB.getIndexStats()
print(stats)


##
## Set parameters for the pipeline
##

query_expansion = [1]  # , 2, 3]
rerank = [True, "rrf", False]  # , "rrf", False]  # True, "rrf", False
prepost_context = [False]
background_reversed = [False]
num_ref_lim = [3]  # [1, 2, 4, 6]


##
## Run pipeline for all parameter permutations. in parallel
##


# Define parameter space
parameters = list(
    itertools.product(
        query_expansion,
        rerank,
        prepost_context,
        background_reversed,
        num_ref_lim,
    )
)

# Define number of queries to sample
n_sample_queries = 100

# Manually adjust this path to the desired output directory with the variable parameter
write_to_dir = "./pipe_results/miniBiosQA/rerank1/"
# make sure to create the directory before running the script
os.makedirs(write_to_dir, exist_ok=True)


# Iterate over multiple LLMs
for LLM_NAME in ["llama3.1:latest"]:
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
