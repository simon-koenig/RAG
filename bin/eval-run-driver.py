# Test of rag evaluation
# Imports
import concurrent.futures
import sys
import time
from functools import partial
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

import logging

from csv_helpers import (
    get_csv_files_from_dir,
)
from drivers import eval_single_pipe_result

# Get pipe results file names
pipe_results_dir = "./parallel_100_rows_pipe/miniBiosQA"
pipe_results_file_names = get_csv_files_from_dir(pipe_results_dir)
# Define directory for eval results
eval_results_dir = "./parallel_100_rows_eval/miniBiosQA"

# Define eval params
method = "all"
evaluator = "llm_judge"
n_worker = 8

# Time the evaluation
start = time.time()
# Loop over all pipe results files to conduct evaluation.
# Create a partial function to pass the fixed parameters to the helper function
# While varying the pipe results file name
partial_helper_vary_input_file = partial(
    eval_single_pipe_result,  # Function to call for single pipe result evaluation
    pipe_results_dir=pipe_results_dir,
    eval_results_dir=eval_results_dir,
    method=method,
    evaluator=evaluator,
    slice_for_dev=100,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
    futures = []
    for pipe_results_file_name in pipe_results_file_names[:]:  # Slice for dev
        if "quExp1" in pipe_results_file_name:
            logging.info(f"Submitting task for {pipe_results_file_name}")
            future = executor.submit(
                partial_helper_vary_input_file, pipe_results_file_name
            )
            futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logging.error(f"Generated an exception: {exc}")


# for pipe_results_file_name in pipe_results_file_names[:]:  # Slice for dev
#     if "quExp1" in pipe_results_file_name:
#         if "numRefLim2" not in pipe_results_file_name:
#             partial_helper_vary_input_file(pipe_results_file_name)

# print(f"Generated an exception: {exc}")


end = time.time()
print(
    f"Time taken for eval of : {end - start} seconds for eval of {len(pipe_results_file_names)} files.\n"
    f"with 100 rows per file.\n"
    f"Method {method} and evaluator {evaluator}.\n"
)
