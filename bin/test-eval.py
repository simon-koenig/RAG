# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

import os

from csv_helpers import read_pipe_results_from_csv
from evaluate import evaluate
from plot_helpers import plot_histogram

sys.path.append("./dev/")
sys.path.append("./src/")

import logging

from csv_helpers import (
    get_csv_files_from_dir,
)
from drivers import eval_single_pipe_result

# Dataset
dataset = "miniWiki"  # miniBiosQA, miniWiki
# Define eval params
method = (
    "quExp"  # Which parameter setting to evaluate, stored by folder name, pool = all
)
# Define evaluator
evaluator = "ROUGE-1"  # Which evaluator to use
select = "correctness"  # Which selection method to use, either "correctness", "cr", "ar", "faithfulness", "all"
n_worker = 4

# Get pipe results file names
pipe_results_dir = f"./parallel_100_rows_pipe/{dataset}/{method}/"
pipe_results_file_names = get_csv_files_from_dir(pipe_results_dir)
# Define directory for eval results
eval_results_dir = f"./parallel_100_rows_eval/{dataset}/{method}/{evaluator}/"
# Create the directory if it does not exist
os.makedirs(eval_results_dir, exist_ok=True)
