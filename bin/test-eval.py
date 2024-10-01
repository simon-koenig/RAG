# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")

from csv_helpers import read_pipe_results_from_csv
from evaluate import evaluate
from plot_helpers import plot_histogram

## Define eval params
rag_settings = "prepostF_rerankT"
method = "correctness"
evaluator = "sem_similarity"
csv_file_path = f"./rag_results/{rag_settings}.csv"
## Read pipe results from CSV
pipe_results = read_pipe_results_from_csv(filename=csv_file_path)
# for elem in pipe_results:
#    pprint(elem)

## Evaluate pipe results

eval_results = evaluate(pipe_results[:10], method=method, evaluator=evaluator)
# pprint(eval_results)


# Plot histogram of eval results
results = list(eval_results.values())
print(results)
print(sum(results) / len(results))
# Save the plot to a file
# file_path = f"./rag_results/plots/{rag_settings}_{method}_{evaluator}.png"
# Plot the histogram
# plot_histogram(data=results, num_bins=10, file_path=file_path)
