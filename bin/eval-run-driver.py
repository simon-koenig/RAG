# Test of rag evaluation
# Imports
import concurrent.futures
import sys
import time
from functools import partial
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

from csv_helpers import (
    get_csv_files_from_dir,
)
from drivers import eval_single_pipe_result

# Get pipe results file names
pipe_results_dir = "./parallel_100_rows_pipe/miniWiki"
pipe_results_file_names = get_csv_files_from_dir(pipe_results_dir)
# Define directory for eval results
eval_results_dir = "./parallel_100_rows_eval/miniWiki"

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

with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
    for pipe_results_file_name in pipe_results_file_names[:1]:  # Slice for dev
        if "quExp1" in pipe_results_file_name:
            executor.submit(partial_helper_vary_input_file, pipe_results_file_name)

# print(f"Generated an exception: {exc}")


end = time.time()
print(
    f"Time taken for eval of : {end - start} seconds for eval of {len(pipe_results_file_names)} files.\n"
    f"with 10 rows per file.\n"
    f"Method {method} and evaluator {evaluator}.\n"
)
