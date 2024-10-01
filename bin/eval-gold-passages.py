# Test of rag evaluation
# Imports
import sys
import time
from pprint import pprint

import pandas as pd

sys.path.append("./dev/")
sys.path.append("./src/")

from csv_helpers import (
    get_csv_files_from_dir,
    read_pipe_results_from_csv,
    write_eval_results_to_csv,
)

# Get pipe results file names
pipe_results_dir = "./parallel_100_rows_pipe/miniBiosQA"
pipe_results_file_names = get_csv_files_from_dir(pipe_results_dir)

# Define eval params
method = "all"
evaluator = "sem_similarity"

# Define dataframe to hold the parameter confiurations and the gold passage matches
context_matches_df = pd.DataFrame(columns=["param_settings", "matches", "sum_matches"])
# Time the evaluation
start = time.time()
# Loop over all pipe results files to conduct evaluation
for pipe_results_file_name in pipe_results_file_names[:]:  # Slice for dev
    # Only look at files with quExp1_rerank1_cExpFalse_backRevFalse_numRef
    pipe_results_file = f"{pipe_results_dir}/{pipe_results_file_name}"
    print(pipe_results_file)
    # Get param settings from the file name
    param_settings = pipe_results_file_name.split("_")

    # Test print results
    # for elem in pipe_results:
    #    pprint(elem)

    # Evaluate pipe results
    slice_for_dev = 5  # Slice for dev
    df = pd.read_csv(pipe_results_file)

    # Keep only the 'contexts_ids' and 'goldPassages' columns
    # Calcualte matches between the ids in the two columns
    df["contexts_ids"] = df["contexts_ids"].apply(
        lambda row: list(map(int, row.split(", "))) if row else []
    )
    print(df["goldPassages"])
    df["goldPassages"] = df["goldPassages"].apply(
        lambda row: list(map(int, row.split(", "))) if row else []
    )
    # Calculate matches between the ids in the two columns
    matches = df.apply(
        lambda row: len(set(row["contexts_ids"]).intersection(row["goldPassages"])),
        axis=1,
    )
    # Add the matches to the DataFrame
    df = df[["contexts_ids", "goldPassages"]]
    # df["matches"] = matches

    # Write the eval results to a csv file
    # eval_results_dir = "./parallel_100_rows_eval"
    # Add param settings and matches to the DataFrame
    # print(f"Param settings: {param_settings}")
    # print(f"Matches \n: {matches.array}")
    # Add the param settings and matches row per row to the DataFrame
    context_matches_df = pd.concat(
        [
            context_matches_df,
            pd.DataFrame(
                [
                    {
                        "param_settings": param_settings,
                        "matches": matches.array,
                        "sum_matches": matches.sum(),
                    }
                ],
            ),
        ],
        ignore_index=True,
    )


end = time.time()
print("Context matches: \n")
print(context_matches_df)
print(f"Execution time: {end - start}")
