# Test of rag evaluation
# Imports
import sys
from pprint import pprint

sys.path.append("./dev/")
sys.path.append("./src/")
from csv_write import write_context_relevance_to_csv
from dataset_helpers import DatasetHelpers
from pipe import RagPipe
from vector_store import VectorStore

# Define API ENDPOINTS
LLM_URL = "http://10.103.251.104:8040/v1"
LLM_NAME = "llama3"
MARQO_URL = "http://10.103.251.104:8882"
MARQO_URL_GPU = "http://10.103.251.104:8880"


# Load QM queries
datasetHelpers = DatasetHelpers()
corpus_list, queries, ground_truths, goldPassages = datasetHelpers.loadMiniBioasq()


# Load the VectorStore on marqo 2.9 with cpu
documentDB = VectorStore(MARQO_URL)  # Connect to marqo client via python API
print(documentDB.getIndexes())  # Print all indexes


# Connect to the miniwikiindex / ait-qm / minibios-qa
documentDB.connectIndex("minibios-qa")
print(documentDB.getIndexStats())  # Print index stats


# Load the VectorStore on marqo 2.10 with gpu
documentDB_GPU = VectorStore(MARQO_URL_GPU)  # Connect to marqo client via python API
documentDB_GPU.connectIndex("minibios-qa-gpu")
print(documentDB_GPU.getIndexes())  # Print all indexes
# Index Docs and measure time
import time

import seaborn as sns

cpu_index_times = []
gpu_index_times = []
maxDocsRange = [10, 50, 100, 200, 500, 1000]
for maxDocs in maxDocsRange:
    start = time.time()
    documentDB.indexDocuments(corpus_list, maxDocs)  # Add documents to the index
    end = time.time()
    ## Time for indexing 100 documents on marqo with cpu
    print(f"Time for indexing {maxDocs} documents: {end - start} seconds")
    cpu_index_times.append(end - start)

    start = time.time()
    documentDB_GPU.indexDocuments(corpus_list, maxDocs)  # Add documents to the index
    end = time.time()
    ## Time for indexing 100 documents on marqo with cpu
    print(f"Time for indexing {maxDocs} documents: {end - start} seconds")
    gpu_index_times.append(end - start)

print(cpu_index_times)
print(gpu_index_times)

import matplotlib.pyplot as plt
import seaborn as sns

# Plotting the line plot
plt.plot(maxDocsRange, cpu_index_times, label="CPU")
plt.plot(maxDocsRange, gpu_index_times, label="GPU")

# Adding labels and title
plt.xlabel("maxDocs")
plt.ylabel("Time (seconds)")
plt.title("CPU Index Times vs GPU Index Times")

# Adding legend
plt.legend()

# Saving the plot as a PDF
plt.savefig(
    "/home/simon/master_project/software/chatbot/llm_param_dev/bin/gpu-cpu-index-comp.pdf"
)

documentDB.deleteIndex("minibios-qa")
documentDB_GPU.deleteIndex("minibios-qa-gpu")
