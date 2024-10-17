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


from csv_helpers import (
    get_csv_files_from_dir,
    read_pipe_results_from_csv,
    write_eval_results_to_csv,
)
from evaluate import evaluate

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3.1:latest"
LLM_70B_NAME = "llama3.1:70b"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


##
## Load file and define file path to write to
##
pipe_results_dir = "./pipe_results/triad/"
pipe_results_file_name = (
    "quExp1_rerankTrue_cExpTrue_backRevFalse_numRefLim4_"
    "modelflax-sentence-embeddings_all_datasets_v4_mpnet-base_"
    "LLM{llama3.1:latest}_.csv"
)

# llms = "llama3.1:latest", "llama3.1:70b", "llama3-chatqa:8b", "mixtral:latest"
llms = ["llama3.1:latest"]
for pipe_results_file_name in get_csv_files_from_dir(pipe_results_dir):
    pipe_results_file = f"{pipe_results_dir}/{pipe_results_file_name}"
    print(f"Running eval on {pipe_results_file_name}.")

    ##
    ## Load pipe results from csv file
    ##

    pipe_results = read_pipe_results_from_csv(pipe_results_file)

    ##
    ## Run evaluation
    ##

    slice_for_dev = len(pipe_results)
    select = "all"
    evaluator = "ROUGE-1"
    # Time
    start = time.time()
    eval_results = evaluate(
        rag_elements=pipe_results,
        select=select,
        evaluator=evaluator,
    )
    end = time.time()
    print(f"Time taken: {end - start} for eval of {slice_for_dev} rows.")

    # pprint(eval_results)

    ##
    ## Write results of a single pipe run to a csv file
    ##
    eval_results_dir = "./eval_results/triad/"
    os.makedirs(eval_results_dir, exist_ok=True)
    write_eval_results_to_csv(
        eval_results=eval_results,
        eval_results_dir=eval_results_dir,
        pipe_results_file=pipe_results_file,
        select=select,
        evaluator=evaluator,
        slice_for_dev=slice_for_dev,
        write_context=False,
    )

    # Done message
    print(f"Done! Eval results written to {eval_results_dir}/{pipe_results_file_name}.")


# evaluationrun()
