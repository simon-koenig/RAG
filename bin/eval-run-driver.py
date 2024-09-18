# Test of rag evaluation
# Imports
import sys
import time
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

from csv_helpers import (
    get_csv_files_from_dir,
    read_pipe_results_from_csv,
    write_eval_results_to_csv,
)
from evaluate import eval

# Get pipe results file names
pipe_results_dir = "./parallel_100_rows_pipe"
pipe_results_file_names = get_csv_files_from_dir(pipe_results_dir)

# Define eval params
method = "all"
evaluator = "sem_similarity"

# Time the evaluation
start = time.time()
# Loop over all pipe results files to conduct evaluation
for pipe_results_file_name in pipe_results_file_names[:]:  # Slice for dev
    # Only look at files with quExp1_rerank1_cExpFalse_backRevFalse_numRef
    if "quExp1_" not in pipe_results_file_name:
        continue
    pipe_results_file = f"{pipe_results_dir}/{pipe_results_file_name}"
    print(pipe_results_file)

    # Read pipe results from CSV
    pipe_results = read_pipe_results_from_csv(filename=pipe_results_file)

    # Test print results
    # for elem in pipe_results:
    #    pprint(elem)

    # Evaluate pipe results
    slice_for_dev = 100  # Slice for dev
    eval_results = eval(
        pipe_results[:slice_for_dev], method=method, evaluator=evaluator
    )
    pprint(eval_results)

    # Write the eval results to a csv file
    eval_results_dir = "./parallel_100_rows_eval"

    # Write results of a single pipe run to a csv file
    write_eval_results_to_csv(
        eval_results=eval_results,
        eval_results_dir=eval_results_dir,
        pipe_results_file=pipe_results_file,
        method=method,
        evaluator=evaluator,
        slice_for_dev=slice_for_dev,
    )


end = time.time()
print(
    f"Time taken for eval of : {end - start} seconds for eval of {len(pipe_results_file_names[1:])} pipe run.\n"
    f"Method {method} and evaluator {evaluator}.\n"
)
