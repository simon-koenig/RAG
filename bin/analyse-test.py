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
eval_results_dir = "./parallel_100_rows_eval"
eval_results_file_names = get_csv_files_from_dir(eval_results_dir)

# Loop over all eval results files
configs = []
corr_results = []
for filename in eval_results_file_names[:6]:  # Iterate over all eval results files
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
    corr_results.append(mean)
    # Get the configs
    config = filename.split("_")
    # Append the configs to the list
    configs.append(config)

print(configs)
print(corr_results)
# Save the plot to a file
# file_path = f"./rag_results/plots/{rag_settings}_{method}_{evaluator}.png"
# Plot the histogram
# plot_histogram(data=results, num_bins=10, file_path=file_path)
