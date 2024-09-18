# Test of rag evaluation
# Imports
import sys
import threading
import time
from functools import partial
from pprint import pprint

import numpy as np

sys.path.append("./dev/")
sys.path.append("./src/")

import concurrent.futures
import itertools

from csv_helpers import write_pipe_results_to_csv
from dataset_helpers import DatasetHelpers
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

##
## Set parameters for the pipeline
##

query_expansion = [3]
rerank = [True, "rrf", False]  # , "rrf", False]  # True, "rrf", False
prepost_context = [False, True]
background_reversed = [False, True]  # [False, True]
num_ref_lim = [2, 4, 6]  # [1, 2, 4, 6]

# Calculate the dimensionality of the parameter space
parameter_count = np.prod(
    [
        len(query_expansion),
        len(rerank),
        len(prepost_context),
        len(background_reversed),
        len(num_ref_lim),
    ]
)


def pipe_parameter_run(parameters, n_slice_rag_elements, write_to_dir):
    # Unpack parameter permuation
    (
        query_expansion_val,
        rerank_val,
        prepost_context_val,
        background_reversed_val,
        num_ref_lim_val,
    ) = parameters

    # Load the RagPipe
    pipe = RagPipe()
    pipe.connectVectorStore(documentDB)
    pipe.connectLLM(LLM_URL, LLM_70B_NAME)

    # Set pipeline configurations
    pipe.setConfigs(
        lang="EN",
        query_expansion=query_expansion_val,
        rerank=rerank_val,
        prepost_context=prepost_context_val,
        background_reversed=background_reversed_val,
        search_ref_lex=4,
        search_ref_sem=4,
        num_ref_lim=num_ref_lim_val,
        model_temp=0.0,
        answer_token_num=50,
    )

    # Run pipeline
    # With slice of rag elements for dev
    pipe.run(
        questions=queries[:n_slice_rag_elements],
        ground_truths=ground_truths[:n_slice_rag_elements],
        goldPassagesIds=goldPassages[:n_slice_rag_elements],
    )

    print("Pipeline run completed.")
    # Print results
    # for elem in pipe.rag_elements:
    #    pprint(elem)

    # Write results to csv file
    # Build the csv file path for the current parameter setting
    csv_file_path = write_to_dir
    csv_file_path += f"quExp{query_expansion_val}_"
    csv_file_path += f"rerank{rerank_val}_"
    csv_file_path += f"cExp{prepost_context_val}_"
    csv_file_path += f"backRev{background_reversed_val}_"
    csv_file_path += f"numRefLim{num_ref_lim_val}_"
    csv_file_path += ".csv"

    write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)


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

n_worker = 8
n_slice_rag_elements = 100
write_to_dir = "./parallel_100_rows_pipe/"

# Define helper function
dev_helper_func = partial(
    pipe_parameter_run,
    n_slice_rag_elements=n_slice_rag_elements,
    write_to_dir=write_to_dir,
)
# Run the pipeline in parallel
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
    executor.map(dev_helper_func, parameters)

print(f"Execution with {n_worker} workers")
end = time.time()
temp = end - start

print(f"Execution time: {temp}")
print("Parameter count: ", parameter_count)
print("Execution time per parameter setting: ", temp / parameter_count)
print(
    "Execution time per parameter setting per worker: ",
    temp / parameter_count / n_worker,
)
print("Number of rag elements : 100")

# Save the results to a file
# with open("parallel_test_dir/execution_times.txt", "w") as file:
#     # Write header
#     file.write("Execution times (nWorker, time (sec))\n")
#     for item in times:
#         file.write(f"{item}\n")
