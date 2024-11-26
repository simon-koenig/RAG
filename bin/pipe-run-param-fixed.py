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

from csv_helpers import write_pipe_results_to_csv
from dataset_helpers import DatasetHelpers
from drivers import pipe_single_setting_run
from rag_pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3.1:latest"  # "llama3.1:latest",  "llama3.1:70b", "llama3-chatqa:8b", "mixtral:latest"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"

# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = datasetHelpers.loadMiniBiosqa()

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

query_expansion = [1, 2, 3]
rerank = [False, True, "rrf"]  # , "rrf", False]  # True, "rrf", False
prepost_context = [False, True]
background_reversed = [False]  # , True]
num_ref_lim = [1, 2, 3, 4]  # [1, 2, 4, 6]

# Calculate the dimensionality of the parameter space

##
## Run pipeline for all for all parameter setting, while keeping another one fixed
##

# Define parameter space
# query_expansion, rerank, prepost_context, background_reversed, num_ref_lim
fixed_param = [1, True, True]
# Put in dict
parameters = {
    "query_expansion": fixed_param[0],
    "rerank": fixed_param[1],
    "prepost_context": fixed_param[2],
    "num_ref_lim": num_ref_lim,
}


### The one parameter to vary: LLMS
LLMS = ["llama3.1:latest", "llama3.1:70b", "command:r", "mixtral:latest"]
for num_sources in num_ref_lim:
    # Load the RagPipe
    pipe = RagPipe()
    pipe.connectVectorStore(documentDB)
    pipe.connectLLM(LLM_URL, LLM_NAME)

    # Set pipeline configurations
    pipe.setConfigs(
        lang="EN",
        query_expansion=parameters["query_expansion"],
        rerank=parameters["rerank"],
        prepost_context=parameters["prepost_context"],
        background_reversed=parameters["background_reversed"],
        search_ref_lex=8,
        search_ref_sem=8,
        num_ref_lim=num_sources,
        model_temp=0.0,
        answer_token_num=50,
    )

    # Run pipeline
    # With slice of rag elements for dev
    n_sample_queries = 10
    n_queries = len(queries)
    k = n_queries // n_sample_queries
    queries = queries[::k][:n_sample_queries]
    ground_truths = ground_truths[::k][:n_sample_queries]
    if goldPassages is not None:
        goldPassages = goldPassages[::k][:n_sample_queries]

    pipe.run(
        questions=queries,
        ground_truths=ground_truths,
        goldPassagesIds=goldPassages,
    )

    print("Pipeline run completed.")
    # Print results
    # for elem in pipe.rag_elements:
    #    pprint(elem)

    ##
    ##  Filename determines:  parameter setting.
    ##

    csv_file_path = "test_dir"
    csv_file_path += ".csv"

    write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)


# Run pipeline for all parameter permutations
n_sample_queries = 50
write_to_dir = "./parallel_100_rows_pipe/miniBiosQA/"
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
n_worker = 8
# Run the pipeline in parallel


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Parallel
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
    futures = [executor.submit(dev_helper_func, param) for param in parameters]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logging.error(f"Generated an exception: {exc}")
    print(f"Execution with {n_worker} workers")


end = time.time()
temp = end - start
# outfile.write(f"Execution with {n_worker} workers took {temp} seconds\n")


# Sequential
# for param in parameters:
#   dev_helper_func(param)


print(f"Execution time: {temp}")
