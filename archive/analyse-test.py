import sys

import pandas as pd

sys.path.append("./dev/")
sys.path.append("./src/")

from csv_helpers import (
    get_csv_files_from_dir,
    read_pipe_results_from_csv,
    write_correctness_to_csv,
)
from plot_helpers import plot_histogram

# Get eval results file names
eval_results_dir = "./parallel_100_rows_eval/miniWiki"
eval_results_file_names = get_csv_files_from_dir(eval_results_dir)

# Loop over all eval results files
# Get column names which are the config paramters and the mean correctness or some other metric
columns = [
    "queExp",
    "rerank",
    "cExp",
    "backRev",
    "numRefLim",
    "metric",
    "evaluator",
] + ["MeanCorrectness"]
results_df = pd.DataFrame(columns=columns)

for filename in eval_results_file_names[:]:  # Iterate over all eval results files
    # Filter files: Only look at files with quExp1_rerank1_cExp*_backRevFalse_numRef4

    print(filename)
    # first_file = f"{eval_results_dir}/{eval_results_file_names[1]}"  # Slice for dev
    file = eval_results_dir + "/" + filename  # Slice for dev
    # Read eval results from CSV
    eval_results_df = pd.read_csv(file)
    print(eval_results_df.head())
    correctness_values = eval_results_df["Correct"].dropna().tolist()
    print(f"Correctness values: {correctness_values}")
    mean = sum(correctness_values) / len(correctness_values)
    print(f"Mean correctness: {mean}")
    # Append the correctness values to the list
    config = filename.split("_")
    print(f" Config: {config}")
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "queExp": config[0],
                        "rerank": config[1],
                        "cExp": config[2],
                        "backRev": config[3],
                        "numRefLim": config[4],
                        "metric": config[5],
                        "evaluator": config[6],
                        "MeanCorrectness": mean,
                    }
                ]
            ),
        ]
    )


# Extract rows where 'backRev' is fixed and 'evaluator' is fixed
fixed_backrev = "backRevFalse"
fixed_evaluator = "llm"  # Replace with the actual evaluator you want to fix
fixed_queExp = "quExp1"

filtered_results_df = results_df[
    (results_df["evaluator"] == fixed_evaluator)
    & (results_df["queExp"] == fixed_queExp)
    & (results_df["MeanCorrectness"] > 0.1)
]
print(f"Filtered results: {filtered_results_df}")

# Get whole row of df row where MeanCorrectness is max and min
max_mean_correctness = filtered_results_df["MeanCorrectness"].max()
max_mean_correctness_row = filtered_results_df[
    filtered_results_df["MeanCorrectness"] == max_mean_correctness
]
print(f"Max mean correctness: {max_mean_correctness_row}")
min_mean_correctness = filtered_results_df["MeanCorrectness"].min()
min_mean_correctness_row = filtered_results_df[
    filtered_results_df["MeanCorrectness"] == min_mean_correctness
]
print(f"Min mean correctness: {min_mean_correctness_row}")
# Save the plot to a file
# file_path = f"./rag_results/plots/{rag_settings}_{method}_{evaluator}.png"
# Plot the histogram
# plot_histogram(data=results, num_bins=10, file_path=file_path)
