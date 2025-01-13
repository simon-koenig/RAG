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
from pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3.1:latest"
LLM_70B_NAME = "llama3.1:70b"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"

##
## Load Dataset
##


# datasetHelpers = DatasetHelpers()
# corpus_list, queries, ground_truths, goldPassages = (
#     datasetHelpers.loadMiniWiki()
# )  # Mini Wiki

datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = (
    datasetHelpers.loadMiniBiosqa()
)  # Mini Bios

##
## Load the VectorStore
##

documentDB = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes
# documentDB.connectIndex("miniwiki-gpu")  # Connect to the miniwiki
documentDB.connectIndex("minibios-qa-gpu")  # Connect to the minibio
stats = documentDB.getIndexStats()
print(stats)

##
## Set parameters for the pipeline
##

query_expansion = 1
rerank = False  # , "rrf", False]  # , "rrf", False]  # True, "rrf", False
prepost_context = False  # , True]
background_reversed = False  #  , True]  # [False, True]
num_ref_lim = 5  # [1, 2, 4, 6]
model = "flax-sentence-embeddings_all_datasets_v4_mpnet-base"
LLM_NAME = "llama3.1:latest"
# LLM_NAME =   # "llama3.1:70b", "llama3.1:latest", "llama3-quatqa:8b", "mixtral:latest"
##
## Load the RagPipe
##

pipe = RagPipe()
pipe.connectVectorStore(documentDB)
pipe.connectLLM(LLM_URL, LLM_NAME)

pipe.setConfigs(
    lang="EN",
    query_expansion=query_expansion,
    rerank=rerank,
    prepost_context=prepost_context,
    background_reversed=background_reversed,
    search_ref_lex=8,
    search_ref_sem=8,
    num_ref_lim=num_ref_lim,
    model_temp=0.0,
    answer_token_num=50,
)

# Time
start = time.time()

##
## Run pipeline
##

# run with ith slice of rag elements for dev

n_slice_rag_elements = 10
write_to_dir = "./pipe_results/triad/"
os.makedirs(write_to_dir, exist_ok=True)

# Generate artificial questions and ground truths
n_artificial_queries = 401
art_queries = [
    f"What has happended in the year {i}?" for i in range(n_artificial_queries)
]
art_ground_truths = [
    f"Event {i} has happended in the year {i}." for i in range(n_artificial_queries)
]


pipe.run(
    questions=art_queries,
    ground_truths=art_ground_truths,
)

print("Pipeline run completed.")
# Print results
# for elem in pipe.rag_elements:
#    pprint(elem)

# Write results to csv file
# Build the csv file path for the current parameter setting

csv_file_path = write_to_dir
csv_file_path += f"quExp{query_expansion}_"
csv_file_path += f"rerank{rerank}_"
csv_file_path += f"cExp{prepost_context}_"
csv_file_path += f"backRev{background_reversed}_"
csv_file_path += f"numRefLim{num_ref_lim}_"
csv_file_path += f"LLM{LLM_NAME}_"
csv_file_path += ".csv"

write_pipe_results_to_csv(pipe.rag_elements, csv_file_path)
end = time.time()
print(" Dataset used is miniBiosQA")
print(f"Time: {end - start} seconds." + "\n")
print(f"Time: {np.round((end - start) / 60, 2)} minutes " + "\n")
print(f"Time: {np.round((end - start) / 3600, 2)} hours " + "\n")
print(f"Model: {model}")
print(f"LLM: {LLM_NAME}")
